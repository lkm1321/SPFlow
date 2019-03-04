from spn.structure.Base import Sum, Product, assign_ids
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.Validity import is_valid
from spn.structure.leaves.parametric.Parametric import Gaussian, Parametric, Bernoulli
from spn.algorithms.TransformStructure import Prune as prune
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up, get_nodes_by_type
from spn.io.Graphics import plot_spn
from collections import deque

import numpy as np
import os
import multiprocessing

class ImageRegion:
    def __init__(self, x_max, y_max, x_min=0, y_min=0, sum_nodes = None, x_max_entire_region = None, y_max_entire_region = None): 

        assert x_max - x_min > 0 and y_max - y_min > 0, "ImageRegion: max min assertion failed"

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self._decomposed = False

        if x_max_entire_region:
            self.x_max_entire_region = x_max_entire_region
        else:
            self.x_max_entire_region = x_max

        if y_max_entire_region:
            self.y_max_entire_region = y_max_entire_region
        else:
            self.y_max_entire_region = y_max


        # If sum_node is passed, it's from a recursive call. 
        if sum_nodes:
            self.sum_nodes = sum_nodes
            # print(len(sum_nodes))
        else:
            new_sum_node = Sum()
            new_sum_node.scope = [ self.get_index(i, j) for i in range(0, self.x_max_entire_region) for j in range(0, self.y_max_entire_region) ]
            self.sum_nodes = [new_sum_node]

        # self.horizontal = horizontal

    def get_index(self, x, y):
        return int(np.ravel_multi_index( (x, y), (self.x_max_entire_region, self.y_max_entire_region) ))

    def get_four_index(self, x_min, y_min, x_max, y_max):
        return int(np.ravel_multi_index( (x_min, y_min, x_max-1, y_max-1), \
                                        (self.x_max_entire_region, self.y_max_entire_region, \
                                         self.x_max_entire_region, self.y_max_entire_region) ))
    def spawn_left_decomposition(self, decomposition_point, num_sum_per_region, min_size = 1):
        left_scope = set( [self.get_index(i, j) for i in range(self.x_min, decomposition_point) 
                                                for j in range(self.y_min, self.y_max)] )
        # This looks crappy, but we need to do this because [Sum()] * num_sum_per_region regards all the sum nodes as being identical. 
        child_sums_left = self.get_nodes_for_scope(left_scope, num_sum_per_region, min_size)
        child_left = ImageRegion( decomposition_point, 
                                self.y_max, 
                                x_min = self.x_min, 
                                y_min = self.y_min, 
                                sum_nodes = child_sums_left, 
                                x_max_entire_region = self.x_max_entire_region, 
                                y_max_entire_region = self.y_max_entire_region)

        return child_left, left_scope

    def spawn_right_decomposition(self, decomposition_point, num_sum_per_region, min_size = 1):
        right_scope = set( [self.get_index(i, j) for i in range(decomposition_point, self.x_max) 
                                                 for j in range(self.y_min, self.y_max)] )
        child_sums_right = self.get_nodes_for_scope(right_scope, num_sum_per_region, min_size)

        child_right = ImageRegion( self.x_max, 
                                    self.y_max, 
                                    x_min = decomposition_point, 
                                    y_min = self.y_min, 
                                    sum_nodes = child_sums_right, 
                                    x_max_entire_region = self.x_max_entire_region, 
                                    y_max_entire_region = self.y_max_entire_region)

        return child_right, right_scope

    def spawn_top_decomposition(self, decomposition_point, num_sum_per_region, min_size = 1):
        top_scope = set( [self.get_index(i, j) for i in range(self.x_min, self.x_max) 
                                                for j in range(decomposition_point, self.y_max)] )
        child_sums_top = self.get_nodes_for_scope(top_scope, num_sum_per_region, min_size)
        # Top
        child_top = ImageRegion(self.x_max, 
                                self.y_max, 
                                x_min = self.x_min, 
                                y_min = decomposition_point, 
                                sum_nodes = child_sums_top, 
                                x_max_entire_region = self.x_max_entire_region, 
                                y_max_entire_region = self.y_max_entire_region)

        return child_top, top_scope

    def spawn_bottom_decomposition(self, decomposition_point, num_sum_per_region, min_size=1):
        bottom_scope = set( [self.get_index(i, j) for i in range(self.x_min, self.x_max) 
                                                for j in range(self.y_min, decomposition_point)] )
        child_sums_bottom = self.get_nodes_for_scope(bottom_scope, num_sum_per_region, min_size)
        # Bottom
        child_bottom = ImageRegion(self.x_max, 
                                    decomposition_point, 
                                    x_min = self.x_min, 
                                    y_min = self.y_min, 
                                    sum_nodes = child_sums_bottom, 
                                    x_max_entire_region = self.x_max_entire_region, 
                                    y_max_entire_region = self.y_max_entire_region) 

        return child_bottom, bottom_scope

    def get_nodes_for_scope(self, scope, num_sum_per_region, min_size): 
        sum_nodes = []

        if len(scope) > min_size: 
            for _ in range(num_sum_per_region): 
                new_child_sum = Sum()
                new_child_sum.scope = scope
                sum_nodes.append(new_child_sum)
        else: 
            # print('adding input nodes')
            for idx in scope: 
                new_child = Bernoulli(p = 0.5, scope = idx)
                sum_nodes.append(new_child)

        return sum_nodes

    ## Binary decomposition down to min_size
    def decompose(self, min_size, num_sum_per_region, decompose_once=False, dropout_prob=0.0, region_dict=None, max_decompositions=-1, depth=None): 

        # Prevent calling decompose twice on a region. 
        if self._decomposed: 
            # print('Decompose called twice')
            return None
        else:
            self._decomposed = True

        # This is the child we want to append to my sum nodes. 
        product_nodes =[]

        # If we're below coarse res, go down to finer decomposition
        # TODO: Might have to check for an AND condition. 
        # DONE: check for AND condition.
        # if self.x_max - self.x_min <= min_size: 
        #     actual_min_size_hrz = 1
        # else:
        #     actual_min_size_hrz = min_size

        # if self.y_max - self.y_min <= min_size: 
        #     actual_min_size_vrt = 1
        # else:
        #     actual_min_size_vrt = min_size

        if self.x_max - self.x_min <= min_size and self.y_max - self.y_min <= min_size:
            actual_min_size_hrz = 1
            actual_min_size_vrt = 1
        else: 
            actual_min_size_hrz = min_size
            actual_min_size_vrt = min_size

        # If not decompose_once, return all the splits in child_regions for external queue. 
        child_regions = []

        # Horizontal cuts. 
        # Range from x_min + actual_min_size + x_max - actual_min_size INCLUSIVE (hence the + 1)

        # if max_decompositions > 0: #and ( (self.x_max - self.x_min) // actual_min_size_hrz > 1 ):
        #     # possible_decomposition_points = np.random.randint(1, (self.x_max - self.x_min) // actual_min_size_hrz, size = max_decompositions)
        #     # possible_decomposition_points = self.x_min + actual_min_size_hrz * np.unique(possible_decomposition_points)
        # possible_decomposition_points = np.unique(np.random.randint(self.x_min + 1,
        #                                                             self.x_max, 
        #                                                             max_decompositions))
        #     possible_decomposition_points = possible_decomposition_points.tolist()
        # else: 
        possible_decomposition_points = range(self.x_min + actual_min_size_hrz, self.x_max - actual_min_size_hrz + 1, 1)

        if len(possible_decomposition_points) > 0:
            decomposition_points = np.unique(np.random.choice(possible_decomposition_points, size=max_decompositions))
        else: 
            decomposition_points = []
        # decomposition_points = possible_decomposition_points
        # len(possible_decomposition_points)
        for decomposition_point in decomposition_points:

            # Default behavior. Don't consult a dictionary. 
            if region_dict is None: 
                child_left, left_scope = self.spawn_left_decomposition(
                                            decomposition_point, 
                                            num_sum_per_region, 
                                            actual_min_size_hrz)

                child_right, right_scope = self.spawn_right_decomposition(
                                                decomposition_point,
                                                num_sum_per_region,
                                                actual_min_size_hrz)
            # Behavior with the dictionary. 
            else: 
                # Check if proposed left split already exists. 
                left_region_index = self.get_four_index(self.x_min, 
                                                        self.y_min, 
                                                        decomposition_point, 
                                                        self.y_max)

                if left_region_index in region_dict.keys(): 
                    child_left = region_dict[left_region_index]
                    assert child_left.x_min == self.x_min
                    assert child_left.x_max == decomposition_point
                    assert child_left.y_min == self.y_min
                    assert child_left.y_max == self.y_max

                    left_scope = set( child_left.sum_nodes[0].scope )
                else:
                    child_left, left_scope = self.spawn_left_decomposition(
                                                decomposition_point,
                                                num_sum_per_region,
                                                actual_min_size_hrz)
                    region_dict[left_region_index] = child_left

                # Check if proposed right split already exists. 
                right_region_index = self.get_four_index(decomposition_point, 
                                                        self.y_min, 
                                                        self.x_max, 
                                                        self.y_max)

                if right_region_index in region_dict.keys(): 
                    child_right = region_dict[right_region_index]
                    assert child_right.x_min == decomposition_point
                    assert child_right.x_max == self.x_max
                    assert child_right.y_min == self.y_min
                    assert child_right.y_max == self.y_max

                    right_scope = set( child_right.sum_nodes[0].scope )

                else:
                    child_right, right_scope = self.spawn_right_decomposition(
                                                    decomposition_point, 
                                                    num_sum_per_region,
                                                    actual_min_size_hrz)
                    region_dict[right_region_index] = child_right

            product_lr_scope = left_scope | right_scope

            # For this particular decomposition, get all possible combination of sum nodes in L/R regions. 
            for child_sum_left in child_left.sum_nodes:
                for child_sum_right in child_right.sum_nodes: 
                    product_node = Product(children=[child_sum_left, child_sum_right])
                    product_node.scope = product_lr_scope
                    product_nodes.append(product_node)

            # Recursive call, recurse down the network. 
            if not decompose_once:
                child_left.decompose(min_size, num_sum_per_region, region_dict=region_dict, max_decompositions=max_decompositions)
                child_right.decompose(min_size, num_sum_per_region, region_dict=region_dict, max_decompositions=max_decompositions)

            # Append and return for outside queue to manage. 
            else: 
                child_regions.append(child_left)
                child_regions.append(child_right)

        if max_decompositions > 0 and (self.y_max - self.y_min) // actual_min_size_vrt > 1:
            possible_decomposition_points = np.random.randint(1, (self.y_max - self.y_min) // actual_min_size_vrt, max_decompositions)
            possible_decomposition_points = self.y_min + actual_min_size_vrt * np.unique(possible_decomposition_points)
        else: 
            possible_decomposition_points = range(self.y_min + actual_min_size_vrt, self.y_max - actual_min_size_vrt+1, actual_min_size_vrt)

        # Vertical cuts. 
        # Range from x_min + actual_min_size + x_max - actual_min_size INCLUSIVE (hence  the + 1)
        for decomposition_point in possible_decomposition_points: 
            
            if region_dict is None:
                child_bottom, bottom_scope = self.spawn_bottom_decomposition(
                                                decomposition_point, 
                                                num_sum_per_region,
                                                actual_min_size_vrt)

                child_top, top_scope = self.spawn_top_decomposition(
                                            decomposition_point, 
                                            num_sum_per_region,
                                            actual_min_size_vrt)
            else: 
                bottom_region_index = self.get_four_index(self.x_min, 
                                                        self.y_min, 
                                                        self.x_max, 
                                                        decomposition_point)

                if bottom_region_index in region_dict.keys():
                    child_bottom = region_dict[bottom_region_index]
                    assert child_bottom.x_min == self.x_min
                    assert child_bottom.x_max == self.x_max
                    assert child_bottom.y_min == self.y_min
                    assert child_bottom.y_max == decomposition_point

                    bottom_scope = set( child_bottom.sum_nodes[0].scope )
                else:
                    child_bottom, bottom_scope = self.spawn_bottom_decomposition(
                                                    decomposition_point, 
                                                    num_sum_per_region,
                                                    actual_min_size_vrt)

                    region_dict[bottom_region_index] = child_bottom
                    
                    if not decompose_once:
                        child_bottom.decompose(min_size, num_sum_per_region, region_dict=region_dict, max_decompositions=max_decompositions)
                    else:
                        child_regions.append(child_bottom)



                top_region_index = self.get_four_index(self.x_min, 
                                                       decomposition_point, 
                                                        self.x_max, 
                                                        self.y_max)

                if top_region_index in region_dict.keys():
                    child_top = region_dict[top_region_index]
                    assert child_top.x_min == self.x_min
                    assert child_top.x_max == self.x_max
                    assert child_top.y_min == decomposition_point
                    assert child_top.y_max == self.y_max

                    top_scope = set( child_top.sum_nodes[0].scope )
                else:
                    child_top, top_scope = self.spawn_top_decomposition(
                                                    decomposition_point, 
                                                    num_sum_per_region,
                                                    actual_min_size_vrt)

                    region_dict[top_region_index] = child_top
                    if not decompose_once:
                        child_top.decompose(min_size, num_sum_per_region, region_dict=region_dict, max_decompositions=max_decompositions)
                    else:
                        child_regions.append(child_top)

            product_bt_scope = bottom_scope | top_scope

            for child_sum_node_btm in child_bottom.sum_nodes:
                for child_sum_node_top in child_top.sum_nodes: 
                    product_node = Product(children=[child_sum_node_btm, child_sum_node_top])
                    product_node.scope = product_bt_scope
                    product_nodes.append(product_node)


        if len(product_nodes) > 0:
            # Add children to myself. Don't add a single product node. 
            for sum_node in self.sum_nodes:
                sum_node.children.extend(product_nodes)
                num_child = len(sum_node.children)
                sum_node.weights = [ 1.0 / num_child ] * num_child

        if decompose_once:    
            return child_regions
        else:
            return None

def get_dense_spn_nonrecursive_multiproc(imsize, coarse_res, num_sum_per_region, num_components_per_var, cpus = 1):
 
    # if do_multiprocess: 
    #     cpus = multiprocessing.cpu_count() - 1
    # else:
    #     cpus = 1

    def worker(_task_queue):
        # print('Worker up')
        while not _task_queue.empty(): 
            print( _task_queue.qsize() )
            region_to_decompose = _task_queue.get()
            _child_regions = region_to_decompose.decompose(coarse_res, num_sum_per_region, num_components_per_var, decompose_once=True)
            if _child_regions is not None: 
                for _child_region in _child_regions:
                    _task_queue.put_nowait(_child_region)
        # print('Worker down')

    region = ImageRegion(imsize[0], imsize[1])

    regions_queue = multiprocessing.Queue()
    # Initialize job queue.
    regions_queue.put(region)
    
    # Give the other CPUs something to do. 
    worker(regions_queue)

    # regions_queue = deque([region])
    procs = []
    for _ in range(cpus):
        p = multiprocessing.Process(target=worker, args=(regions_queue,))
        procs.append(p)
        p.start()

    for p in procs:
        p.join() 

    # while len(regions_queue) > 0: 
    #     region_to_decompose = regions_queue.get()
    #     # region_to_decompose = regions_queue.popleft()
    #     while len(regions_queue) < cpus:
    #         p = multiprocessing.Process(target=worker, args=(regions_queue, region_to_decompose))

        # child_regions = region_to_decompose.decompose(coarse_res, num_sum_per_region, num_components_per_var, decompose_once=True)
        # if child_regions is not None:
        #     regions_queue.extend(child_regions)

    spn = region.sum_nodes[0]
    assign_ids(spn)
    return spn

def get_dense_spn_nonrecursive(imsize, coarse_res, num_sum_per_region, max_decompositions=-1):
 
    # if do_multiprocess: 
    #     cpus = multiprocessing.cpu_count() - 1
    # else:
    #     cpus = 1
        # print('Worker down')

    region = ImageRegion(imsize[0], imsize[1])

    regions_queue = deque([region])
    # region_dict = None
    region_dict = dict()

    while len(regions_queue) > 0: 
        # region_to_decompose = regions_queue.get()
        # print(len(regions_queue))
        region_to_decompose = regions_queue.popleft()
    #     while len(regions_queue) < cpus:
    #         p = multiprocessing.Process(target=worker, args=(regions_queue, region_to_decompose))

        child_regions = region_to_decompose.decompose(coarse_res, 
                                                    num_sum_per_region, 
                                                    decompose_once=True, 
                                                    region_dict=region_dict, 
                                                    max_decompositions=max_decompositions)

        if child_regions is not None:
            regions_queue.extend(child_regions)

    spn = region.sum_nodes[0]
    assign_ids(spn)
    return spn

def get_dense_spn(imsize, coarse_res, num_sum_per_region, max_decompositions=-1):
    region = ImageRegion(imsize[0], imsize[1])
    region_dict = dict()
    region.decompose(coarse_res, num_sum_per_region, region_dict=region_dict, max_decompositions=max_decompositions)
    spn = region.sum_nodes[0]
    assign_ids(spn)
    # rebuild_scopes_bottom_up(spn)
    # assert is_valid(spn)
    return spn


if __name__=="__main__":
    region = ImageRegion(64, 64)
    region.decompose(8, 20, 10)
    spn = region.sum_nodes[0]
    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)
    print(get_structure_stats(spn))
    print(is_valid(spn))
    # plot_spn(spn, fname='spn.png')

    print(get_structure_stats(prune(spn)))
 
    # print(spn.scope)
    print(is_valid(spn))
    # nodes = get_nodes_by_type(spn)
    # print(type(nodes[12]))
    # print(nodes[12].children)
    # print(spn.weights)
    plot_spn(prune(spn), fname='spn_prune.png')


