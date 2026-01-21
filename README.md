# NBA Predictor

# Description
I often find myself looking through [basketballreference.com ](https://www.basketball-reference.com/) trying to find out what NBA teams would win in a given match up. So I decided to make a NBA game predictor that scrapes the stats from [basketballreference.com ](https://www.basketball-reference.com/) and determines a winner. Using a sklearn machine learning model, the winner is determined by a teams field goal %, rebounds, assists, turnovers, opponent stats, home/away record, and takes into account injuries. Additionally the model also considers the recent form of a team and the severity of a given injury. So an injury to a star means more then an end of the bench guy. Along with that, there is a historic predictor which scrapes stats from any season from 2024-2025 to 1999-2000.

# Built With
This project was built using the following python libraries:
- BeautifulSoup (Web scraping) https://beautiful-soup-4.readthedocs.io/en/latest/
- sklearn (Machine Learning Model) https://scikit-learn.org/stable/
- NumPy (Math functions) https://numpy.org/
- pandas (Data manipulation) https://pandas.pydata.org/

# Usage

### Select a home and away team and click Predict
<img width="1409" height="895" alt="image" src="https://github.com/user-attachments/assets/7092808e-ceed-499c-8693-1933fecc26cf" />

### The injuries of both the home and away team are displayed along with the impact on the win probability 
<img width="1267" height="904" alt="image" src="https://github.com/user-attachments/assets/5367f298-5aec-41aa-a058-ca7be24ca764" />

### Historic Predictor allows you to select two teams from two different seasons
<img width="1532" height="885" alt="image" src="https://github.com/user-attachments/assets/45887be8-b53e-4fa2-9ce4-0fba18e7af37" />

