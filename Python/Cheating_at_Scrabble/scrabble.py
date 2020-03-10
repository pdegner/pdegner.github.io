import sys
letters = sys.argv[1]

def scrabble(letters):
    """Takes an input of up to 7 letters including wildcards and returns possible scrabble words.
    Earns a grade of 90% or better for student 'Patricia Degner'"""
    
    
    letters = letters.lower()
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    scores = {"a": 1, "c": 3, "b": 3, "e": 1, "d": 2, "g": 2, "f": 4, "i": 1, 
              "h": 4, "k": 5, "j": 8, "m": 3, "l": 1, "o": 1, "n": 1, "q": 10, 
              "p": 3, "s": 1, "r": 1, "u": 1, "t": 1, "w": 4, "v": 4, "y": 4, 
              "x": 8, "z": 10, "*":0, "?":0
             }
    import itertools
    from score_word import score_word

    #Import the scrabble dictionary
    with open("sowpods.txt","r") as infile:
        raw_input = infile.readlines()
    WordData = [datum.strip('\n') for datum in raw_input]
    
    #Check for valid input:
    ##Check length of input
    if len(letters) <2 or len(letters)>7:
        raise Exception("InputError: You must use 2 to 7 letters.") 
    ##Check to ensure valid letters
    for L in letters:
        if L not in scores:
            raise Exception("InputError: You must enter only letters or the wildcards '*' and '?'.")
    ##Check no more than 2 wildcards
    wildcards = 0
    for L in letters:
        if L == "*" or L == "?":
            wildcards += 1
        if wildcards > 2:
            raise Exception("InputError: You may not have more than 2 wildcards.")
            
    #Create a function that obtains a list of valid words
    def word_getter(letters):
        """Create permutations of the letters, stored in tuples, which are stored in a set"""
        #I just want you to know I spent about 2 hours working on a recursion that does this
        #Then I discovered itertools
        #I'm never getting that 2 hours back am I?
        a = [l for l in letters]
        FirstSet = set()
        for i in range(1,len(letters)):  
            for Tup in list(itertools.permutations(a,i+1)):
                FirstSet.add(Tup)

        #Write a function to convert tuples to a string
        #Use it to create a NewSet of strings from FirstSet, each of which is a permutation of letters
        def TupleConvert(tup):
            """Converts tuples of characters or strings into a single string"""
            new_tuple = "".join(tup)
            return new_tuple
        NewSet = set()
        for tup in FirstSet:
             NewSet.add(TupleConvert(tup))

        #Save only words in the scrabble dictionary to FinalSet
        FinalSet = set()
        for word in NewSet:
            if word.upper() in WordData:
                FinalSet.add(word)
        return FinalSet
            
    #Call function that finds the value of the words and stores them in a list of tuples
    
    #If there is not a wildcard, create a sorted list of tuples and return the elements
    if "*" not in letters and "?" not in letters:
        Final_list = [(score_word(word, letters), word) for word in word_getter(letters)]
        Final_list.sort(reverse = True)
        for result in Final_list:
            print(result)
        print("Total number of words:",len(Final_list))
        
    #If the wildcard is a question mark:
    elif "?" in letters:
        letters = letters.replace("?","*")
        scrabble(letters)
    
    #Now deal with the wildcard
    else:
        #Make a function that replaces multiple wildcards and stores the results in a list
        #These objects are labeled recursion because this function was originally recursive
        #I ultimately had to write out the recursion but didn't change the names
        def wc_replacement(word_with_2wc):
            """Replaces the wildcards in words with more than 1 wildcard
            Returns a list of potential words"""
            recursion_set = [word_with_2wc.replace("*",l,1) for l in alphabet]
            recursion_set = set(recursion_set)
            recursion_list = set()
            for wc_word in recursion_set:
                add_to_set = [wc_word.replace("*",l) for l in alphabet]
                for word in add_to_set:
                    recursion_list.add(word)
            recursion_list = list(recursion_list)
            return recursion_list
        
        #If there is one wildcard:
        if letters.count("*") == 1:
            recursion_list = [letters.replace("*",l) for l in alphabet]
        #If there are two wildcards, use the function
        else:
            recursion_list = wc_replacement(letters)
        
        #Obtain a list of sets of valid words, and some empty sets
        recursion_result = [word_getter(recur) for recur in recursion_list]
        
        #Remove empty sets and save to a list
        wc_list = []
        for result in recursion_result:
            if len(result) >0:
                wc_list.append(result)
        
        #Turn the list of sets into a list of lists. Add each element of each sublist to wc_list2. 
        #This turns my data into a list of strings that are valid scrabble words
        wc_list2 = []
        for Set in wc_list:
            Set = list(Set)
            for i in range(0,len(Set)):
                wc_list2.append(Set[i])
        
        #Turn to a set, then back to a list to remove duplicates
        wc_list2 = set(wc_list2)
        wc_list2 = list(wc_list2)
        
        #Use function to calculate the value of the word sans wildcard

        #Create a list sorted list of tuples and return the elements
        Final_wc_list = [(score_word(word, letters), word) for word in wc_list2]
        Final_wc_list.sort(reverse = True)
        
        for result in Final_wc_list:
            print(result)
        print("Total number of words:",len(Final_wc_list))

scrabble(letters)