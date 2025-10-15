# you can add imports but you should not rely on libraries that are not already provided in "requirements.txt #
from collections import deque
from heapq import heappush, heappop
import heapq
import numpy as np

# for weighted a star, function and self-testing
import time
import random 


class TextbookStack(object):
    """A class that tracks the"""

    def __init__(self, initial_order, initial_orientations):
        assert len(initial_order) == len(initial_orientations)
        self.num_books = len(initial_order)

        for i, a in enumerate(initial_orientations):
            assert i in initial_order
            assert a == 1 or a == 0

        self.order = np.array(initial_order)
        self.orientations = np.array(initial_orientations)

    def flip_stack(self, position):
        assert position <= self.num_books

        self.order[:position] = self.order[:position][::-1]
        self.orientations[:position] = np.abs(self.orientations[:position] - 1)[
            ::-1
        ]

    def check_ordered(self):
        for idx, front_matter in enumerate(self.orientations):
            if (idx != self.order[idx]) or (front_matter != 1):
                return False

        return True

    def copy(self):
        return TextbookStack(self.order, self.orientations)

    def __eq__(self, other):
        assert isinstance(
            other, TextbookStack
        ), "equality comparison can only ba made with other __TextbookStacks__"
        return all(self.order == other.order) and all(
            self.orientations == other.orientations
        )

    def __str__(self):
        return f"TextbookStack:\n\torder: {self.order}\n\torientations:{self.orientations}"


def apply_sequence(stack, sequence):
    new_stack = stack.copy()
    for flip in sequence:
        new_stack.flip_stack(flip)
    return new_stack

# helper function for a star search
# admissiable heuristic - number of books not in place or not facing front
# def heuristic(stack):
    
#     h=0
#     for i in range(stack.num_books):
#         if stack.order[i] != i or stack.orientations[i] != 1:
#             h += 1
    
#     return h

def a_star_search(stack):
    
#     # --- v ADD YOUR CODE HERE v --- #
    flip_sequence = []
    
    # # initialize
    # # stack is the initial gistate
    # start_key = str(stack)
    # # open set is the list of nodes to explore
    # open_set = [stack]
    # # flip sequence is the list of flips to get to the goal state
    # flip_sequence = []
    # # dictionary to track each state to the sequence of flips used to reach it
    # node_path = {str(stack): []}
    # # set of visited nodes
    # closed_set = set()

    # heap implementation for efficiency
    # using a heap for open set to optimize retrieval of lowest f score
    
    # stack in string version
    start_key = str(stack)
    # PQ for open set
    open_heap = []  
    # start node into open_heap
    heapq.heappush(open_heap, (0, 0, stack))  # (f_score, counter, node)
    # to track each state to the sequence of flips used to reach it
    node_path = {start_key: []}
    # set of visited nodes
    closed_set = set()
    # to break ties in heap
    counter = 0  
    
    # replacer for helper function, inline heurisitic
    h0=0
    for i in range(stack.num_books):
        if stack.order[i] != i or stack.orientations[i] != 1:
            h0 += 1
    
    # f, g, h scores    
    # f = g+h
    # cost from source node to node
    g = {str(stack): 0}
    # estimated cost from source to goal through node
    h = {str(stack): h0}
    # total cost from source to goal node
    f = {str(stack): g[str(stack)] + h[str(stack)]}

    # while there are nodes to explore
    while open_heap:

        # get node in open set with lowest f score (and remove from open), set as current
        f_curr, _, curr = heapq.heappop(open_heap)
        curr_key = str(curr)

        # if curr_key not in node_path:
        if curr_key not in node_path:
            # keep current key empty
            node_path[curr_key] = []
        # if curr_key not in g:
        if curr_key not in g:
            # cost from source to curr is length of path to curr
            g[curr_key] = len(node_path[curr_key])
            
            # inline heuristic
            h_neighbor = 0
            for j in range(curr.num_books):
                if (curr.order[j] != j or curr.orientations[j] != 1):
                    h_neighbor += 1

            # calculate h and f for curr
            h[curr_key] = h_neighbor
            f[curr_key] = g[curr_key] + h[curr_key]

        # if current node is goal state, return flip sequence
        if curr.check_ordered():
            return node_path[curr_key]
        
        # mark node as visitied
        closed_set.add(curr_key)
        
        # explore neighbors
        # for each flip position, copy the stack, do the flip, get the string key
        for i in range(1, curr.num_books + 1):
            neighbor = curr.copy()
            neighbor.flip_stack(i)
            neighbor_key = str(neighbor)

            # if neighbor is in closed list, skip
            if neighbor_key in closed_set:
                continue
            
            # new cost to reach neighbor
            tentative_g = g[curr_key] + 1
            
            # inline heuristic
            h_neighbor = 0
            for j in range(curr.num_books):
                if (curr.order[j] != j or curr.orientations[j] != 1):
                    h_neighbor += 1

            # caluclate g, h, f for neighbor
            # record the path
            g[neighbor_key] = tentative_g
            h[neighbor_key] = h_neighbor
            f[neighbor_key] = tentative_g + h_neighbor
            node_path[neighbor_key] = node_path[curr_key] + [i]
            counter += 1 # increment counter for tie-breaking in heap
            # push with total cost and total tie-breaker
            heapq.heappush(open_heap, (f[neighbor_key], counter, neighbor))

    return flip_sequence
    # return flip_sequence

    # if no solution found
    # return []
#     # ---------------------------- #


def weighted_a_star_search(stack, epsilon=None, N=1):
    # Weighted A* is extra credit
    # choosing option 2 : total running time for each m 

    # --- v ADD YOUR CODE HERE v --- #


    return flip_sequence

    # ---------------------------- #

##  GIVEN TEST FOR A STAR SEARCH

if __name__ == "__main__":
    test = TextbookStack(initial_order=[3, 2, 1, 0], initial_orientations=[0, 0, 0, 0])
    output_sequence = a_star_search(test)
    correct_sequence = int(output_sequence == [4])

    new_stack = apply_sequence(test, output_sequence)
    stack_ordered = new_stack.check_ordered()

    print(f"Stack is {'' if stack_ordered else 'not '}ordered")
    print(f"Comparing output to expected traces  - \t{'PASSED' if correct_sequence else 'FAILED'}")
   
    