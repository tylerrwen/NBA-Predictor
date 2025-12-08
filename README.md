# NBA Predictor

# Description
I often find myself looking through [basketballreference.com ](https://www.basketball-reference.com/) trying to find out what NBA teams would win in a given match up. So I decided to make a NBA game predictor that scrapes the stats from [basketballreference.com ](https://www.basketball-reference.com/) and determines a winner. Using a sklearn machine learning model, the winner is determined by a teams field goal %, rebounds, assists, turnovers, opponent stats, home/away record, and takes into account injuries. Additionally the model also considers the recent form of a team and the severity of a given injury. So an injury to a star means more then an end of the bench guy.

# Built With
This project was built using the following python libraries:
- BeautifulSoup (Web scraping) https://beautiful-soup-4.readthedocs.io/en/latest/
- sklearn (Machine Learning Model) https://scikit-learn.org/stable/
- NumPy (Math functions) https://numpy.org/
- pandas (Data manipulation) https://pandas.pydata.org/

# Usage

### Select a home and away team and click Predict
<img width="1197" height="763" alt="image" src="https://github.com/user-attachments/assets/d0ee1607-63db-48be-87db-efc458e55695" />

### The injuries of both the home and away team are displayed along with the impact on the win probability 
<img width="1052" height="842" alt="image" src="https://github.com/user-attachments/assets/2e3422e2-83e0-4283-bf2e-85ef6f9a7cd1" />
