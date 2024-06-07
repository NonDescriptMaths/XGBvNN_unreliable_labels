import slune

if  __name__ == "__main__":
    to_search_info_rank = {
        'benchmark': [
            'fully_labelled', 
            'missing_labels'
            ],
        'model': [
            'neural_net',
            'xgboost',
            ],
    }
    grid_info_rank = slune.searchers.SearcherGrid(to_search_info_rank)

    script_path = 'exp.py'
    template_path = 'compute_spec.sh'
    slune.sbatchit(script_path, template_path, grid_info_rank)
