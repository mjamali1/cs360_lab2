# you can add imports but you should not rely on libraries that are not already provided in "requirements.txt #
from collections import deque
from heapq import heappush, heappop
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
def heuristic(stack):
    
    h=0
    for i in range(stack.num_books):
        if stack.order[i] != i or stack.orientations[i] != 1:
            h += 1
    
    return h



def a_star_search(stack):
    
#     # --- v ADD YOUR CODE HERE v --- #
    
    # initialize
    # stack is the initial gistate
    open_set = [stack]
    # flip sequence is the list of flips to get to the goal state
    flip_sequence = []
    # dictionary to track each state to the sequence of flips used to reach it
    node_path = {str(stack): []}
    # set of visited nodes
    closed_set = set()

    # f = g+h
    # cost from source node to node
    g = {str(stack): 0}
    # estimated cost from source to goal through node
    h = {str(stack): heuristic(stack)}
    # total cost from source to goal node
    f = {str(stack): g[str(stack)] + h[str(stack)]}

    # while there are nodes to explore
    while open_set:

        # get node with lowest f value, keeping curr as string and int
        # line 100 disclaimer - used outside sources for guidance
        # my original line and subsequent attempts would not compile
        curr = min(open_set, key=lambda stack: f[str(stack)]) 
        curr_key = str(curr)

        # if current node is goal state, return flip sequence
        if curr.check_ordered():
            flip_sequence = node_path[curr_key]
            return flip_sequence
        
        # mark node as visitied
        open_set.remove(curr)
        closed_set.add(curr_key)

        # explore neighbors
        # for each neighbor of current node
        for i in range (1, stack.num_books + 1):
            neighbor = curr.copy()
            neighbor.flip_stack(i)
            neighbor_key = str(neighbor)

            # if neighbor is in closed list, skip
            if neighbor_key in closed_set:
                continue

            # cost from source to neighbor
            tentative_g = g[str(curr)] + 1

            # if neighbor is not in open set, add it
            if neighbor not in open_set:
                open_set.append(neighbor)
            # else if tentative g is greater than g of neighbor, skip
            elif tentative_g >= g.get(str(neighbor), float('inf')):
                continue

        # record f score to neighbor
        g[str(neighbor)] = tentative_g
        h[str(neighbor)] = heuristic(neighbor)
        f[str(neighbor)] = g[str(neighbor)] + h[str(neighbor)]

        # record path to neighbor
        node_path[neighbor_key] = node_path[curr_key] + [i]

    # return least cost path from start to end
    return flip_sequence

    # if no solution found
    return []
#     # ---------------------------- #


def weighted_a_star_search(stack, epsilon=None, N=1):
    # Weighted A* is extra credit
    # choosing option 2 : total running time for each m 

    # new f score = g + epsilon * h
    flip_sequence = []
    # # --- v ADD YOUR CODE HERE v --- #
    # for i in range(N):
    #     start = time.time()

    #     open_set = [stack]
    #     node_path = {str(stack): []}
    #     closed_set = set()

    #     # f, g, h scores
    #     g = {str(stack): 0}
    #     h = {str(stack): heuristic(stack)}
    #     f = {str(stack): g[str(stack)] + (epsilon * h[str(stack)])}

    #     # very similar to a star search
    #     while open_set:
    #         curr = min(open_set, key=lambda stack: f[str(stack)])
    #         curr_key = str(curr)

    #         if curr.check_ordered():
    #             flip_sequence = node_path[curr_key]
    #             return flip_sequence # solution is found
            
    #         open_set.remove(curr)
    #         closed_set.add(curr_key)

    #         # check neighbors
    #         for n in range(1, stack.num_books + 1):
    #             neighbor = curr.copy()
    #             neighbor.flip_stack(n)
    #             neighbor_key = str(neighbor)

    #             if neighbor_key in closed_set:
    #                 continue

    #             tentative_g = g[str(curr)] + 1

    #             if neighbor not in open_set:
    #                 open_set.append(neighbor)
    #             elif tentative_g >= g.get(str(neighbor), float('inf')):
    #                 continue

    #             g[str(neighbor)] = tentative_g
    #             h[str(neighbor)] = heuristic(neighbor)
    #             f[str(neighbor)] = g[str(neighbor)] + (epsilon * h[str(neighbor)])

    #             node_path[neighbor_key] = node_path[curr_key] + [n]

    #     end = time.time()
    #     runtime = end - start
    #     print(f"Runtime for iteration {i+1}: {runtime} seconds")


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

