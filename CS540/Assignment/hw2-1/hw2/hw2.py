import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)



def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()

    for letter in range(65, 91):  # ASCII values for A to Z are 65 to 90
        X[chr(letter)] = 0
        
    with open (filename,encoding='utf-8') as f:

        #The contents in this file
        text = f.read()

        for l in text:
            l = l.upper() #Ignore the cases

            if l >= 'A' and l <= 'Z':
                if l in X: 
                    X[l] += 1
                else:
                    X[l] = 1    
    f.close()

    print("Q1")
    for key in sorted(X.keys()):
        print(key, X[key])
                        
    return X





def q2_solve(filename):
    total_counts = shred(filename)
    X1 = total_counts.get('A', 0)

    (e,s) = get_parameter_vectors()
    e1 = e[0]
    s1 = s[0]

    X1_log_e1 = X1 * math.log(e1)
    X1_log_s1 = X1 * math.log(s1)

    print("Q2")
    print(f"{X1_log_e1:.4f}")
    print(f"{X1_log_s1:.4f}")



def q3_solve(filename, eprior, sprior):
    total_counts = shred(filename)
    (e,s) = get_parameter_vectors()

    #Here I actually compute the log P(Y = y)
    Fenglish = math.log(eprior)
    Fspanish = math.log(sprior)

    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        Xi = total_counts.get(letter, 0)

        letter_index = ord(letter) - ord('A')

        Fenglish += Xi * math.log(e[letter_index])
        Fspanish += Xi * math.log(s[letter_index])

    print("Q3")
    print(f"{Fenglish:.4f}")
    print(f"{Fspanish:.4f}")    

    return (Fenglish, Fspanish)




    
# Compute P(Y = English | X)   
def q4_solve(filename, eprior, sprior):
    q3result = q3_solve(filename, eprior, sprior)
    q4result = 0.0000

    Fenglish = q3result[0]
    Fspanish = q3result[1]

    if Fspanish - Fenglish >= 100:
        q4result = 0.0000
    elif Fspanish - Fenglish <= -100:
        q4result = 1.0000
    else:
        q4result = 1 / (1 + math.exp(Fspanish - Fenglish))    

    print("Q4")
    print(f"{q4result:.4f}")




# Main program
if __name__ == "__main__":

    filename = sys.argv[1]
    english_prior = float(sys.argv[2])
    spanish_prior = float(sys.argv[3])

    shred(filename)
    q2_solve(filename)
    q3_solve(filename, english_prior, spanish_prior)
    q4_solve(filename, english_prior, spanish_prior)