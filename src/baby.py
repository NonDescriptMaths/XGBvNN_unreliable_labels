from xgb import get_xgb
from train import train
from naive_data import naive_get_data




def run():
    # Get data
    data = naive_get_data()
    # Train model
    model = get_xgb()
    train(model, data)