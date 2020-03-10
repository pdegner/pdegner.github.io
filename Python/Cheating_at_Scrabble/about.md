# Cheating at Scrabble

The code for this can be found here: https://github.com/pdegner/pdegner.github.io/tree/master/Python/Cheating_at_Scrabble

"scrabble.py" is a Python program that takes a Scrabble rack as a command-line argument and prints all "valid Scrabble English" words that can be constructed from that rack, along with their Scrabble scores sorted by score, and the total number of words you can make. 

"score_word.py" is a function in a separate module. I imported this function into my main solution code.

"snowpods.txt" contains a list of all the valid Scrabble words, and thier score. 

Some rules:
* You may input 2-7 characters
* You may use \* or ? as wildcards (the blank tile in Scrabble, worth 0 points)
* You may input the same letter more than once (e.g. aaackee)
* The letter will not be used more than once unless it appears more than once in your tiles
