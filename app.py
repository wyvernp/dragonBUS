import yaml
from data import *

def main():
    # Load the configuration file
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # Get the 30-minute usage data
    model = get_usage_model_data(config)
    predicted_usage = predict_next_24_hours(config, model)
    print(predicted_usage)
    import matplotlib.pyplot as plt

    # Plot the predicted usage
    time = predicted_usage['time']
    consumption = predicted_usage['predicted_consumption']
    plt.plot(time, consumption)
    plt.xlabel('Time')
    plt.ylabel('Usage')
    plt.title('Predicted Usage for Next 24 Hours')
    plt.show()
main()





