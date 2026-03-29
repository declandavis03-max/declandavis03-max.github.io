# NBA-PPG-Predictor
Built a Python-based NBA Points per Game (ppg) predicator using live web data such as PPG, Defensive Rating (DRtg) from NBA basketball-reference, all while factoring location of the game.

## Table of Contents

- [Research Question](#Research-Question)
- [ Data Pipeline and Data](#Data-Pipeline-and-Data)
- [Website Used for Data](#Websites-Used-for-Data)
- [Creation of Formula](#Creation-of-Formula)
- [Example Prediction](#Example-Prediction)
- [Overall Takeaways](#Overall-Takeaways)
- [Limitations and Next Steps](#Limitations-and-Next-Steps)
- [Final Statement](#Final-Statement)

# Research Question
Using online data from Sports Reference, Can there be a model that accurately predicts an NBA players points per game on any given night during the NBA season?

# Data Pipeline and Data

### Data Pipeline
WIth Sports Reference Data, the goal is create a points per game predictor using these steps

```
Websites Used for Data --> Creation of Forumla --> Data Collection and Building Functions --> --> Overall Insights
```

### Websites Used for Data
Found Player and Team Statistics for the model. 
    
    -Defensive/Offesnive Team Statistics:
    https://www.basketball-reference.com/leagues/NBA_2026.html
    
    -Individual Player Statistics:
    https://www.basketball-reference.com/leagues/NBA_2026_per_game.html 

Tools Used
    Python | Pandas (pd)


# Creation of Formula

### Prediction Formula

    1. Home Court Factor:
        Appilies a home court factor on points scored depending on where the selted player is playing.
        Home: +1 Points
        Away: -1 Points 

    2. Defensive Rating Adjustment:
        Adjusts the score by comparing opponent's defensive rating compared to league average DRtg, all based on player share of their team's points. 
        (Opp DRtg - League Avg DRtg) * (Player PPG / Team PPG)

    3. Final Rounded Forecast:
    Sums the base player's PPG and the DRtg adjustment, as well as Game location to include a final prediction.

### Final Formula
### Predicted_PPG = Current_PPG - (DRtg_ADJ) +- (Home Court Factor)

# Data Collection and Building of Functions
   
    1. Defined User Parameters:
        Added guided user parameters for the user to provide the model with the players name, 
        his team (abbreviations if needed via dictionary), and location of the game (Home or Away). 

        Abbreviations Below:

![Abbrev Dictionary](images/Abbrev_dictionary.png)

        Main Function within Code with Input for Paramters:
![Main Function](images/main_function.png)
        

    2.  In Depth, Live Web Scraping:  From basektball reference
    (Team PPG, Indidiual PPG, Team Defensive Rating)

## Example Prediction
![Example Test](images/example_test.png)

    
# Overall Takeaways 
    1.Context > Raw Metrics

While Defensive Rating and proportional scoring share significantly influence predicted PPG, they do not fully capture real-world performance. Player output is heavily dependent on context such as:

    - Usage rate
    - Game pace
    - Matchup-specific defensive schemes
    - Teammate availability
**Key insight: Quantitative metrics alone are insufficient without situational context.**

    2. Model Simplicity vs. Real-World Complexity

The model demonstrates that a relatively simple formula using (Player PPG, Team PPG, Team DRtg, can generate reasonable directional predictions. However, basketball performance is non-linear and dynamic, meaning:
    - Small inputs can have unpredictable effects
    - External variables (fatigue, injuries, rotations) are not captured
    
**Key insight: Simple models are useful for baseline predictions, but struggle with real-world variability.**

    3. Value of Modular Code Design
Breaking the project into modular Python functions --> Improved readability, Made debugging easier, Allowed for flexible updates to the model

**Key insight: The proces of building modular functions has been very benefiical for my overall experience on the project**

# Limitations 
- Limited Feature Set:
The model only incorporates a small number of variables (PPG, Team PPG, Defensive Rating, and location). Key metrics such as usage rate, effective field goal percentage (eFG%), and minutes played are not included, which limits predictive accuracy.

- Web Scraping Reliability:
The model depends on live data scraping from Basketball Reference. Any changes to the website structure or delays in data updates can break the pipeline or introduce inconsistencies.

- Small Sample Bias (Per-Game Stats):
Per-game statistics can be misleading, especially for players with limited minutes or small sample sizes, leading to unstable or inflated predictions.

# Final Statement

While the model provides a structured approach to predicting PPG, it highlights that basketball performance is influenced by a wide range of dynamic and contextual factors that extend beyond basic statistical inputs. Additionally, this project served as my first experience building modular functions in Python, and was a great introduction to how powerful and flexible Python can be for data analysis and modeling.




