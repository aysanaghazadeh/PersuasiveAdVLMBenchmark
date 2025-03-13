from Reasoning.retrieval import Retrieval
from configs.evaluation_config import get_args

if __name__ == '__main__':
    args = get_args()
    retrieval = Retrieval(args)
    retrieval.get_all_results()