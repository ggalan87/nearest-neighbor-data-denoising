from pl_bolts.models.mnist_module import *


def cli_main():
    # args
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = LitMNIST.add_model_specific_args(parser)
    args = parser.parse_args()

    # model
    model = LitMNIST(**vars(args))

    # training
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

    trainer.test()


if __name__ == "__main__":  # pragma: no cover
    cli_main()
