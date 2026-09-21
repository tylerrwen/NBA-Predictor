# NBA Predictor

## Description
I often find myself looking through [basketball-reference.com](https://www.basketball-reference.com/) trying to figure out which NBA team would win in a given matchup. So I decided to build an NBA game predictor that scrapes the stats from [basketball-reference.com](https://www.basketball-reference.com/) and determines a winner. Using a scikit-learn machine learning model, the winner is determined by a team's field goal %, rebounds, assists, turnovers, opponent stats, and home/away record, while also accounting for injuries. The model additionally considers the recent form of a team and the severity of a given injury, so an injury to a star means more than one to an end-of-the-bench player. On top of that, there is a historic predictor which scrapes stats from any season between 1999-2000 and 2024-2025.

## Built With
This project was built using the following Python libraries:
- [Flask](https://flask.palletsprojects.com/) (Web app / server)
- [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/) (Web scraping)
- [scikit-learn](https://scikit-learn.org/stable/) (Machine learning model)
- [NumPy](https://numpy.org/) (Math functions)
- [pandas](https://pandas.pydata.org/) (Data manipulation)

## Getting Started

### Prerequisites
- Python 3.11 or newer

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd NBA-Predictor
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install flask beautifulsoup4 scikit-learn numpy pandas
   ```

### Build the data caches
All scraping happens offline. `build_cache.py` is the only script that touches
basketball-reference; the web app serves everything from the caches it writes and
never scrapes during a request. Build them first:
```bash
python build_cache.py
```
This trains the current-season model and caches injuries and key players. See the
options for retraining on specific seasons or pre-warming the historic cache:
```bash
python build_cache.py --current --seasons 2026,2025,2024   # retrain on specific seasons
python build_cache.py --historic --seasons 2024,2023 --teams LAL,BOS
```

### Run the app
```bash
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

## Usage

### Select a home and away team and click Predict
<img width="1409" height="895" alt="image" src="https://github.com/user-attachments/assets/7092808e-ceed-499c-8693-1933fecc26cf" />

### The injuries of both the home and away team are displayed along with the impact on the win probability
<img width="1267" height="904" alt="image" src="https://github.com/user-attachments/assets/5367f298-5aec-41aa-a058-ca7be24ca764" />

### Historic Predictor allows you to select two teams from two different seasons
<img width="1532" height="885" alt="image" src="https://github.com/user-attachments/assets/45887be8-b53e-4fa2-9ce4-0fba18e7af37" />
