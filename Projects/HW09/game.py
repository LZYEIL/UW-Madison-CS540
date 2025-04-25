import random
import copy
import time


class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']


    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        self.depth_count = 3  # Depth limit for minimax



    def run_challenge_test(self):
        """ Set to True if you would like to run gradescope against the challenge AI!
        Leave as False if you would like to run the gradescope tests faster for debugging.
        You can still get full credit with this set to False
        """ 
        return True



    def succ(self, state, piece=None):
        if piece is None:
            piece = self.my_piece

        successors = []
        num_pieces = sum(cell != ' ' for row in state for cell in row)

        if num_pieces < 8:  # Drop phase
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[row][col] = piece
                        successors.append(new_state)
        else:  # Move phase
            for row in range(5):
                for col in range(5):
                    if state[row][col] == piece:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                r, c = row + dr, col + dc
                                if 0 <= r < 5 and 0 <= c < 5 and state[r][c] == ' ':
                                    new_state = copy.deepcopy(state)
                                    new_state[row][col] = ' '
                                    new_state[r][c] = piece
                                    successors.append(new_state)
        return successors



    def make_move(self, state):
        """Select the best move using minimax search"""
        start_time = time.time()
        best_score = float('-inf')
        best_move = None
        drop_phase = sum(cell != ' ' for row in state for cell in row) < 8

        for successor in self.succ(state):
            score = self.min_value(successor, 1)
            if score > best_score:
                best_score = score
                best_move = successor

        # Find the difference between current state and best move
        move_to = None
        move_from = None
        for i in range(5):
            for j in range(5):
                if state[i][j] != best_move[i][j]:
                    if state[i][j] == ' ':
                        move_to = (i, j)
                    else:
                        move_from = (i, j)

        if drop_phase:
            return [move_to]
        else:
            return [move_to, move_from]



    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)




    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece
        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece




    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")



    def game_value(self, state):
        """ Checks the current board status for a win condition
        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.
        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for i in range(2):
            for j in range(2):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]:
                    return 1 if state[i][j] == self.my_piece else -1

        # check / diagonal wins
        for i in range(2):
            for j in range(3, 5):
                if state[i][j] != ' ' and state[i][j] == state[i+1][j-1] == state[i+2][j-2] == state[i+3][j-3]:
                    return 1 if state[i][j] == self.my_piece else -1

        # check box wins
        for i in range(4):
            for j in range(4):
                if state[i][j] != ' ' and state[i][j] == state[i][j+1] == state[i+1][j] == state[i+1][j+1]:
                    return 1 if state[i][j] == self.my_piece else -1

        return 0 # no winner



    def heuristic_game_value(self, state):
        """Evaluate non-terminal states"""
        gval = self.game_value(state)
        if gval != 0:
            return float(gval)

        my_score = 0.0
        opp_score = 0.0

        # Check all possible lines (rows, columns, diagonals)
        for i in range(5):
            # Rows
            row = state[i]
            my_score += self.evaluate_line(row, self.my_piece)
            opp_score += self.evaluate_line(row, self.opp)
            
            # Columns
            col = [state[j][i] for j in range(5)]
            my_score += self.evaluate_line(col, self.my_piece)
            opp_score += self.evaluate_line(col, self.opp)

        # Diagonals
        diag1 = [state[i][i] for i in range(5)]
        diag2 = [state[i][4-i] for i in range(5)]
        my_score += self.evaluate_line(diag1, self.my_piece) + self.evaluate_line(diag2, self.my_piece)
        opp_score += self.evaluate_line(diag1, self.opp) + self.evaluate_line(diag2, self.opp)

        # Normalize scores
        final_score = (my_score - opp_score) / 10.0
        return max(-1.0, min(1.0, final_score))




    def evaluate_line(self, line, piece):
        """Helper function to evaluate a single line"""
        score = 0.0
        for i in range(len(line)-3):
            segment = line[i:i+4]
            count = segment.count(piece)
            empty = segment.count(' ')
            if count == 3 and empty == 1:
                score += 0.8
            elif count == 2 and empty == 2:
                score += 0.4
            elif count == 1 and empty == 3:
                score += 0.1
        return score




    def max_value(self, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state)
        if depth >= self.depth_count:
            return self.heuristic_game_value(state)

        max_score = float('-inf')
        for successor in self.succ(state, self.my_piece):
            score = self.min_value(successor, depth + 1)
            if score > max_score:
                max_score = score
        return max_score




    def min_value(self, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state)
        if depth >= self.depth_count:
            return self.heuristic_game_value(state)

        min_score = float('inf')
        for successor in self.succ(state, self.opp):
            score = self.max_value(successor, depth + 1)
            if score < min_score:
                min_score = score
        return min_score






############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()