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
        --halfSatCellAngularConcentration {parameter_dictionary['halfSatCellAngularConcentration']} \
        --maxCellAngularConcentration {parameter_dictionary['maxCellAngularConcentration']} \
        --halfSatMeanActinFlow {parameter_dictionary['halfSatMeanActinFlow']} \
        --maxMeanActinFlow {parameter_dictionary['maxMeanActinFlow']} \
        --flowScaling {parameter_dictionary['flowScaling']} \
        --polarityDiffusionRate {parameter_dictionary['polarityDiffusionRate']} \
        --actinAdvectionRate {parameter_dictionary['actinAdvectionRate']} \
        --contactAdvectionRate {parameter_dictionary['contactAdvectionRate']} \
        --halfSatMatrixAngularConcentration {parameter_dictionary['halfSatMatrixAngularConcentration']} \
        --maxMatrixAngularConcentration {parameter_dictionary['maxMatrixAngularConcentration']} \
        --cellBodyRadius {parameter_dictionary['cellBodyRadius']} \
        --eccentricity {parameter_dictionary['eccentricity']} \
        --sharpness {parameter_dictionary['sharpness']} \
        --inhibitionStrength {parameter_dictionary['inhibitionStrength']} \
        "

    os.system(run_command)
    return None


if __name__ == '__main__':
    main()
