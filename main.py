""" main work:
    1. parse data from files and preprocessing data
        1) text to vector
        2) balance classes
        3) standardization
    2. model building
    3. model evaluation and prediction

"""
# Author: xxx
# License: xxx

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import Dataset
from model.base import Recommender


def main(random_state=42):

    # 1. get data
    # 1.1 parse file and load data
    user_news_clicks_csv = 'data/user_news_clicks.csv'
    news_text_csv = 'data/news_text.csv'
    dt = Dataset(news_text_csv, user_news_clicks_csv)
    dt.get_data(n_rows=10000,is_balance = True)
    X, y = dt.X, dt.y
    print(f'X.shape: {X.shape}, y: {Counter(y)}')

    # 1.2 split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    # 2. build model
    # rd = Recommender(model_name='DecisionTreeClassifier', random_state=random_state)
    rd = Recommender(model_name='MLP', model_params={'in_dim':X_train.shape[1], 'out_dim': len(Counter(y))},
                     random_state=random_state)
    rd.fit(X_train, y_train)

    # 3. evaluate model
    # rd.test(X_train, y_train)
    rd.test(X_test, y_test)


if __name__ == '__main__':
    main()
