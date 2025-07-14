"""A script to run the simulation based on a set of parameters in a .json file."""
# flake8: noqa: E501
# ^ disabling line length linting warnings.
import argparse
import os
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run simulation using given .json config file.')
    parser.add_argument('--path_to_config', type=str, help='Path to config file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    with open(args.path_to_config, 'r') as file:
        parameter_dictionary = json.load(file)

    # TODO: just build the bash comand using the keys from the parameter dict

    run_command = \
        f"./build/src/main \
        --jobArrayID {parameter_dictionary['jobArrayID']} \
        --superIterationCount {parameter_dictionary['superIterationCount']} \
        --timestepsToRun {parameter_dictionary['timestepsToRun']} \
        --numberOfCells {parameter_dictionary['numberOfCells']} \
        --worldSize {parameter_dictionary['worldSize']} \
        --gridSize {parameter_dictionary['gridSize']} \
        --matrixTurnoverRate {parameter_dictionary['matrixTurnoverRate']} \
        --matrixAdditionRate {parameter_dictionary['matrixAdditionRate']} \
        --thereIsMatrixInteraction {parameter_dictionary['thereIsMatrixInteraction']} \
        --dt {parameter_dictionary['dt']} \
        --cueDiffusionRate {parameter_dictionary['cueDiffusionRate']} \
        --cueKa {parameter_dictionary['cueKa']} \
        --fluctuationAmplitude {parameter_dictionary['fluctuationAmplitude']} \
        --fluctuationTimescale {parameter_dictionary['fluctuationTimescale']} \
        --actinAdvectionRate {parameter_dictionary['actinAdvectionRate']} \
        --matrixAdvectionRate {parameter_dictionary['matrixAdvectionRate']} \
        --collisionAdvectionRate {parameter_dictionary['collisionAdvectionRate']} \
        --maximumSteadyStateActinFlow {parameter_dictionary['maximumSteadyStateActinFlow']} \
        --cellBodyRadius {parameter_dictionary['cellBodyRadius']} \
        --aspectRatio {parameter_dictionary['aspectRatio']} \
        --collisionFlowReductionRate {parameter_dictionary['collisionFlowReductionRate']} \
        "

    os.system(run_command)
    return None


if __name__ == '__main__':
    main()
