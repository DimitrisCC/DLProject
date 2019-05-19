from src.utils import tab_printer
from src.parser import parameter_parser_capsgnn
from src.capsgnn import CapsGNNTrainer


def main():
    """
    Parsing command line parameters, processing graphs, fitting a CapsGNN.
    """
    args = parameter_parser_capsgnn()
    tab_printer(args)
    model = CapsGNNTrainer(args)
    model.fit()
    model.score()
    model.save_predictions()


if __name__ == "__main__":
    main()