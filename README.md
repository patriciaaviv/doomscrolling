# TUM.ai Hackathon: Combating Doomscrolling

## Project Structure
Project is divided in three parts: a python script responsible for the sentiment analysis, 
a browser extension for Chrome browser as a frontend and a flask server connecting the two.
Output of the model is available as .csv file.
The extension is inside the zip file.

## Setup
<code>pip3 install -r requirements.txt</code>

## Result of Analysis
We tested a twitter dataset from 22.04.22 in a csv file as input for the sentiment analysis.
Running main.py results in an updated csv file that contains the tokens of the tweets that input file
in one column and in the other column you get the result of the sentiment analysis..

