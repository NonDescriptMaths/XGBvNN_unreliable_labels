import argparse
import slune
import os
from train import pretrain, train
from xgb import get_xgb
from nn import get_nn
from utils import get_data, get_X_y_labelled, check_preprocessed, get_X_y_unlabelled
from numpy import float32
import wandb

# If running labelled_exp
def labelled_exp(saver, **config):
    check_preprocessed()
    # Get and prepare the data
    train_data, test_data, validate_data = get_data(normalize_data=True)
    X_train, y_train = get_X_y_labelled(train_data)
    # X_val, y_val = get_X_y_labelled(validate_data) Not currently using validation data
    X_test, y_test = get_X_y_labelled(test_data)

    X_train = X_train.to_numpy().astype(float32)
    y_train = y_train.to_numpy().astype(float32)
    X_test = X_test.to_numpy().astype(float32)
    y_test = y_test.to_numpy().astype(float32)

    # Get the model
    if config['model'] == 'xgboost':
        model = get_xgb(lr=config['learning_rate'], n_estimators=config['n_estimators'], max_depth=config['max_depth'])
    elif config['model'] == 'neural_net':
        model = get_nn(
            lr=config['learning_rate'], 
            opt=config['opt'], 
            loss=config['loss'], 
            num_epochs=config['num_epochs'], 
            # batch_size=config['batch_size'],
            # update_ratio=config['update_ratio'],
            num_update_epochs=config['num_update_epochs'],
            MLP_shape=config['MLP_shape'],
            )
    else:
        raise ValueError("Model not recognized.")
    
    # Train the model
    metrics = pretrain(model, X_train, y_train, X_test, y_test)

    # Save results
    # Loop through dictionary and save each metric
    # for i in range(len(metrics['loss']['training'])):
    #     saver.log({'training_loss':metrics['loss']['training'][i]})
    #     saver.log({'validation_loss':metrics['loss']['validation'][i]})
    #     saver.log({'training_auc':metrics['auc']['training'][i]})
    #     saver.log({'validation_auc':metrics['auc']['validation'][i]})
    #     saver.log({'training_acc':metrics['acc']['training'][i]})
    #     saver.log({'validation_acc':metrics['acc']['validation'][i]})
    # saver.save_collated()

    return metrics

# If running missing_labels_exp
def missing_labels_exp(saver, **config):
    check_preprocessed()
    # Get and prepare the data
    train_data, test_data, validate_data = get_data(normalize_data=True)
    X, y = get_X_y_labelled(train_data)
    X_test, y_test = get_X_y_labelled(test_data)

    X_unlabelled, y_unlabelled = get_X_y_unlabelled(train_data)

    X = X.to_numpy().astype(float32)
    y = y.to_numpy().astype(float32)
    X_test = X_test.to_numpy().astype(float32)
    y_test = y_test.to_numpy().astype(float32)
    X_unlabelled = X_unlabelled.to_numpy().astype(float32)
    y_unlabelled = y_unlabelled.to_numpy().astype(float32)

    # Get the model
    if config['model'] == 'xgboost':
        model = get_xgb(lr=config['learning_rate'], n_estimators=config['n_estimators'], max_depth=config['max_depth'])
    elif config['model'] == 'neural_net':
        model = get_nn(
            lr=config['learning_rate'], 
            opt=config['opt'], 
            loss=config['loss'], 
            num_epochs=config['num_epochs'], 
            # batch_size=config['batch_size'],
            # update_ratio=config['update_ratio'],
            num_update_epochs=config['num_update_epochs'],
            MLP_shape=config['MLP_shape'],
            )
    else:
        raise ValueError("Model not recognized.")
    
    # Train the model
    metrics = train(model, X, y, X_test, y_test, X_unlabelled=X_unlabelled, y_unlabelled=y_unlabelled, num_epochs=config['num_epochs'], update_ratio=config['update_ratio'], query_method=config['query_method'], query_alpha=config['query_alpha'], query_K=config['query_K'], query_args=config['query_args'])

    # # Save results
    # saver.log(metrics)
    # saver.save_collated()

    return metrics

def list_to_dict(g):
    def split_at_equals(s):
        return s.split('=')
    
    def maybe_convert_to_numeric(s):
        try:
            if '.' in s:
                return float(s)
            else:
                return int(s)
        except:
            return s
    
    return {str(split_at_equals(s)[0][2:]): maybe_convert_to_numeric(split_at_equals(s)[1]) for s in g}

# Following will be run on the cluster as a job, after being queued by run.py
if  __name__ == "__main__":
    # Parse input from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, help='Name of the benchmark to use', default="labelled_exp")
    parser.add_argument('--model', type=str, help='Model to use', default="xgboost")
    parser.add_argument('--learning_rate', type=float, help='Learning rate to use', default=0.01)
    parser.add_argument('--full_train_every', type=int, help='How often to fully train the model (in epochs)', default=10)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train for', default=200)
    parser.add_argument('--update_ratio', type=float, help='Ratio of old:new data to train on in an active learning iteration', default=0.1)
    parser.add_argument('--batch_size', type=str, help='Batch size to use', default='max')
    parser.add_argument('--opt', type=str, help='Optimizer to use', default="adam")
    parser.add_argument('--loss', type=str, help='Loss function to use', default="logloss")
    parser.add_argument('--num_update_epochs', type=int, help='Number of update epochs to run', default=10)
    parser.add_argument('--MLP_shape', type=str, help='Shape of the MLP', default="128,128")
    parser.add_argument('--query_method', type=str, help='Query method to use', default="")
    parser.add_argument('--query_K', type=int, help='Number of samples to query', default=10)
    parser.add_argument('--query_alpha', type=float, help='Alpha for query method', default=0.1)
    parser.add_argument('--query_args', type=str, help='Additional arguments for query method', default="")

    args = parser.parse_args()

    config = {
        'benchmark': [args.benchmark],
        'model': [args.model],
        'learning_rate': [args.learning_rate],
        'full_train_every': [args.full_train_every],
        'num_epochs': [args.num_epochs],
        'batch_size': [args.batch_size],
        'opt': [args.opt],
        'loss': [args.loss],
        'update_ratio': [args.update_ratio],
        'num_update_epochs': [args.num_update_epochs],
        'MLP_shape': [args.MLP_shape],
        'query_method': [args.query_method],
        'query_K': [args.query_K],
        'query_alpha': [args.query_alpha],
        'query_args': [args.query_args],
    }

    if 'neural_net' in config['model']:
        # config['batch_size'] = [32, 64, 128]
        # config['num_epochs'] = [50, 100, 200]
        config['num_update_epochs'] = [5, 10]
        # config['MLP_shape'] = ['128,128']
    elif 'xgboost' in config['model']:
        # config['learning_rate'] = [0.1, 0.01, 0.001]
        config['n_estimators'] = [100, 200]
        config['max_depth'] = [2, 4]

    if config['benchmark'] != 'labelled_exp':
        config['query_method'] = ['entropy', 'random', 'entrepRE']#, 'margin', 'entrepRBF'] # maybe ignore last 2, or 3?
        config['query_K'] = [10, 50]
        config['query_alpha'] = [0, 0.5, 1]
    

    print("Searching Over: ", config, flush=True)
    grid = slune.searchers.SearcherGrid(config)
    # Get wandb API key, in 'wandb_api_key.txt'
    with open('wandb_api_key.txt', 'r') as f:
        wandb_api_key = f.read()
    os.environ['WANDB_API_KEY'] = wandb_api_key
    # grid.check_existing_runs(slune.get_csv_saver(root_dir='results'))
    for g in grid:
        print("Current params: ", g)
        saver = slune.get_csv_saver(root_dir='results', params=g)
        path = saver.get_current_path()
        print("path: ", path, flush=True)

        g = list_to_dict(g)

        wandb.init(project="ActiveLearning", config=g, name=path)

        # Train the model
        if 'labelled_exp' in g['benchmark']:
            metrics = labelled_exp(saver, **g)
        elif 'missing_labels' in g['benchmark']:
            metrics = missing_labels_exp(saver, **g)
        else:
            raise ValueError("Benchmark not recognized.")

        # print(g[0][12:])
        # if g[0][12:] == 'labelled_exp':
        #     metrics = labelled_exp(saver, **g)
        # elif g[0][12:] == 'missing_labels':
        #     metrics = missing_labels_exp(saver, **g)
        # else:
        #     raise ValueError("Benchmark not recognized.")

        metrics.save(saver)
        
        # Produce and save plots
        # if config['benchmark'] == 'labelled_exp':
        #     plot_labelled_exp(metrics, path)
        # elif config['benchmark'] == 'missing_labels':
        #     plot_missing_labels(metrics, path)
        # TODO: Implement plotting functions