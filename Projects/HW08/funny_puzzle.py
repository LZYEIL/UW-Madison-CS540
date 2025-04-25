import heapq
import copy



def state_check(state):
    """check the format of state, and return corresponding goal state.
       Do NOT edit this function."""
    non_zero_numbers = [n for n in state if n != 0]
    num_tiles = len(non_zero_numbers)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    elif num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    matched_seq = list(range(1, num_tiles + 1))
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    elif not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')
    elif len(set(non_zero_numbers)) != len(non_zero_numbers):
        raise ValueError('State can not have repeated numbers, except 0.')
    elif sorted(non_zero_numbers) != matched_seq:
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                          'and there will be 9-X grids labeled as 0.')
    goal_state = matched_seq
    for _ in range(9 - num_tiles):
        goal_state.append(0)
    return tuple(goal_state)





def get_manhattan_distance(from_state, to_state):
    """
    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    all_dict = {
    0: (0 , 0),
    1: (1 , 0),
    2: (2 , 0),
    3: (0 , 1),
    4: (1 , 1),
    5: (2 , 1),
    6: (0 , 2),
    7: (1 , 2),
    8: (2 , 2)
    }

    distance = 0

    for i in range(len(from_state)):

        if (from_state[i] != 0):
            value = from_state[i]
            tostate_index = to_state.index(value)

            curr_x_pos, curr_y_pos = all_dict.get(i)
            end_x_pos, end_y_pos = all_dict.get(tostate_index)

            l1_norm = abs(curr_x_pos - end_x_pos) + abs(curr_y_pos - end_y_pos)
            distance += l1_norm

    return distance





def naive_heuristic(from_state, to_state):
    """
    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        0 (but experimenting with other constants is encouraged)
    """
    return 0




def sum_of_squares_distance(from_state, to_state):
    """
    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        A scalar that is the sum of squared distances for all tiles
    """
    all_dict = {
    0: (0 , 0),
    1: (1 , 0),
    2: (2 , 0),
    3: (0 , 1),
    4: (1 , 1),
    5: (2 , 1),
    6: (0 , 2),
    7: (1 , 2),
    8: (2 , 2)
    }

    distance = 0

    for i in range(len(from_state)):

        if (from_state[i] != 0):
            value = from_state[i]
            tostate_index = to_state.index(value)

            curr_x_pos, curr_y_pos = all_dict.get(i)
            end_x_pos, end_y_pos = all_dict.get(tostate_index)

            euclid_norm = (curr_x_pos - end_x_pos)**2  + (curr_y_pos - end_y_pos)**2
            distance += euclid_norm

    return distance




def print_succ(state, heuristic=get_manhattan_distance):
    """
    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """

    # given state, check state format and get goal_state.
    goal_state = state_check(state)
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(heuristic(succ_state,goal_state)))





def get_succ(state):
    """
    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle.
    """
    resulting_List = []
    
    # Convert 1D state to 2D grid for easier handling
    grid = []
    for i in range(0, 9, 3):
        grid.append(state[i:i+3])
    
    # Iterate through the 2D grid
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 0:
                # Found empty space, check four directions
                
                # Check up
                if i - 1 >= 0 and grid[i-1][j] != 0:
                    # Create new grid (deep copy)
                    new_grid = [row[:] for row in grid]
                    # Swap with tile above
                    new_grid[i][j], new_grid[i-1][j] = new_grid[i-1][j], new_grid[i][j]
                    # Convert back to 1D and add to results
                    new_state = [cell for row in new_grid for cell in row]
                    resulting_List.append(new_state)
                
                # Check down
                if i + 1 <= 2 and grid[i+1][j] != 0:
                    new_grid = [row[:] for row in grid]
                    new_grid[i][j], new_grid[i+1][j] = new_grid[i+1][j], new_grid[i][j]
                    new_state = [cell for row in new_grid for cell in row]
                    resulting_List.append(new_state)
                
                # Check left
                if j - 1 >= 0 and grid[i][j-1] != 0:
                    new_grid = [row[:] for row in grid]
                    new_grid[i][j], new_grid[i][j-1] = new_grid[i][j-1], new_grid[i][j]
                    new_state = [cell for row in new_grid for cell in row]
                    resulting_List.append(new_state)
                
                # Check right
                if j + 1 <= 2 and grid[i][j+1] != 0:
                    new_grid = [row[:] for row in grid]
                    new_grid[i][j], new_grid[i][j+1] = new_grid[i][j+1], new_grid[i][j]
                    new_state = [cell for row in new_grid for cell in row]
                    resulting_List.append(new_state)
    
    return sorted(resulting_List)







def is_solvable(state):
    """
    Check if a 3x3 sliding tile puzzle state is solvable.
    For puzzles with tiles 1-7, all configurations are solvable.
    For 8-tile puzzles, solvability depends on inversion parity.
    
    Args:
        state: List of 9 integers (0 represents blank spaces).
    
    Returns:
        bool: True if solvable, False otherwise.
    """
    # Count non-zero tiles
    non_zero_tiles = [tile for tile in state if tile != 0]
    num_tiles = len(non_zero_tiles)
    
    # If fewer than 8 tiles, always solvable
    if num_tiles < 8:
        return True
    
    # For 8-tile puzzles, check inversion parity
    if num_tiles == 8:
        # Extract non-zero tiles in their linear order
        tiles_in_order = []
        for i in range(9):
            if state[i] != 0:
                tiles_in_order.append(state[i])
        
        # Count inversions correctly
        inversions = 0
        for i in range(len(tiles_in_order)):
            for j in range(i + 1, len(tiles_in_order)):
                if tiles_in_order[i] > tiles_in_order[j]:
                    inversions += 1
        
        # For 8-tile puzzles, the puzzle is solvable if inversions is even
        return inversions % 2 == 0
    
    return True






def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0], heuristic=get_manhattan_distance):
    """
    INPUT: 
        An initial state (list of length 9)
        
    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along h values, number of moves.
    """
    # Check state format and get goal_state.
    goal_state = state_check(state)
    
    # Check if the puzzle is solvable
    if not is_solvable(state):
        print(False)
        return
    
    # Convert to tuples for hashability
    state = tuple(state)
    goal_state = tuple(goal_state)
    
    # Initialize data structures
    OPEN = []  # Priority queue
    CLOSED = []  # List to store closed nodes
    closed_dict = {}  # For faster lookup: state -> index in CLOSED
    
    # Step 1: Put the start state S on the priority queue OPEN
    h_start = heuristic(state, goal_state)
    g_start = 0
    f_start = g_start + h_start
    # Using the form: heapq.heappush(pq, (cost, state, (g, h, parent_index)))
    heapq.heappush(OPEN, (f_start, state, (g_start, h_start, -1)))  # parent_index -1 means no parent
    
    # Track g values for each state
    g_values = {state: g_start}
    
    max_queue_length = 1
    
    # Main A* loop
    while OPEN:
        # Update max queue length for debugging
        max_queue_length = max(max_queue_length, len(OPEN))
        
        # Step 2: If OPEN is empty, exit with failure - handled by while condition
        
        # Step 3: Remove from OPEN and place on CLOSED a node n for which f(n) is minimum
        node = heapq.heappop(OPEN)
        f_value = node[0]
        current_state = node[1]
        g_value, h_value, parent_idx = node[2]
        
        # If we already found a better path to this state, skip it
        if current_state in g_values and g_value > g_values[current_state]:
            continue
        
        # Step 4: If n is a goal node, exit (recover path by tracing back pointers from n to S)
        if current_state == goal_state:
            # Reconstruct path
            path = []
            current_node = node
            
            while True:
                state_list = list(current_node[1])
                h_val = current_node[2][1]
                g_val = current_node[2][0]
                path.append((state_list, h_val, g_val))
                
                parent_index = current_node[2][2]
                if parent_index == -1:  # We've reached the start state
                    break
                    
                current_node = CLOSED[parent_index]
            
            # Reverse path to get start to goal
            path.reverse()
            
            # Print results
            print(True)
            for state_info in path:
                print(state_info[0], f"h={state_info[1]}", f"moves: {state_info[2]}")
            
            # Print max queue length for debugging
            print(f"Max queue length: {max_queue_length}")
            return
        
        # Add to CLOSED
        CLOSED.append(node)
        closed_dict[current_state] = len(CLOSED) - 1  # Store the index in CLOSED
        
        # Step 5: Expand n, generating all successors and attach to pointers back to n
        successors = get_succ(list(current_state))
        
        for succ in successors:
            succ_state = tuple(succ)
            succ_g = g_value + 1  # One move from current state
            succ_h = heuristic(succ_state, goal_state)
            succ_f = succ_g + succ_h
            
            # If we already found a better path to this successor, skip it
            if succ_state in g_values and succ_g >= g_values[succ_state]:
                continue
                
            # Update the g value for this state
            g_values[succ_state] = succ_g
            
            # Step 5.1 & 5.2: Create successor node and add to OPEN
            # Using the form: (cost, state, (g, h, parent_index))
            succ_node = (succ_f, succ_state, (succ_g, succ_h, len(CLOSED) - 1))
            heapq.heappush(OPEN, succ_node)
    
    # Step 6: Goto 2 - handled by the while loop
    
    # If we exit the loop without finding a solution
    print(False)
    return




# if __name__ == "__main__":

#     solve([4,3,0,5,1,6,7,2,0])




    
    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    # solve([2,5,1,4,0,6,7,0,3], heuristic=get_manhattan_distance)
    # print()
