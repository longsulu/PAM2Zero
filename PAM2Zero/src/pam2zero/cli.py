import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="pam2zero")
    parser.add_argument(
        "command",
        choices=["train", "evaluate", "paper-suite", "factorization", "capacity",
                                           "probability",
                                               "thermodynamics",
                                               "activity",
                                               "self-test"
                                           ])
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sys.argv = ["pam2zero " + args.command] + remaining
    if args.command == "train":
        from pam2zero.experiments.train import main as run
    elif args.command == "evaluate":
        from pam2zero.experiments.evaluate import main as run
    elif args.command == "paper-suite":
        from pam2zero.experiments.paper_suite import final_main as run
    elif args.command == "capacity":
        from pam2zero.analysis.factor_capacity import main as run
    elif args.command == "probability":
        from pam2zero.probability.aligned import main as run
    elif args.command == "activity":
        from pam2zero.activity.training import main as run
    elif args.command == "thermodynamics":
        from pam2zero.analysis.thermodynamics import main as run
    elif args.command == "factorization":
        from pam2zero.experiments.factorization import main as run
    else:
        from pam2zero.probability.aligned import coefficient_self_test, probability_self_test
        coefficient_self_test()
        probability_self_test()
        from pam2zero.activity.validation import self_test
        self_test()
        try:
            import torch
        except ImportError:
            print("PyTorch checks skipped: PyTorch is not installed")
            return
        from pam2zero import config as cfg
        cfg.initialize_device()
        from pam2zero.validation import final_self_test
        final_self_test()
        return
    run()
