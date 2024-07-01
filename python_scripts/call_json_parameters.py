"""A script to run the simulation based on a set of parameters in a .json file."""
import os
import json


def main():
    with open("optimised_fit.json", "r") as file:
        parameter_dictionary = json.load(file)

    run_command = \
        f"./build/src/main \
        --jobArrayID 8 \
        --superIterationCount 20 \
        --timestepsToRun 5760 \
        --numberOfCells 70 \
        --worldSize 2048 \
        --gridSize 32 \
        --cellTypeProportions 0 \
        --matrixAdditionRate {parameter_dictionary["matrixAdditionRate"]} \
        --matrixTurnoverRate {parameter_dictionary["matrixTurnoverRate"]} \
        --cellDepositionSigma {parameter_dictionary["cellDepositionSigma"]} \
        --cellSensationSigma {parameter_dictionary["cellSensationSigma"]} \
        --poissonLambda {parameter_dictionary["poissonLambda"]} \
        --kappa {parameter_dictionary["kappa"]} \
        --matrixKappa {parameter_dictionary["matrixKappa"]} \
        --homotypicInhibition {parameter_dictionary["homotypicInhibition"]} \
        --heterotypicInhibition {parameter_dictionary["heterotypicInhibition"]} \
        --polarityPersistence {parameter_dictionary["polarityPersistence"]} \
        --polarityTurningCoupling {parameter_dictionary["polarityTurningCoupling"]} \
        --flowScaling {parameter_dictionary["flowScaling"]} \
        --flowPolarityCoupling {parameter_dictionary["flowPolarityCoupling"]} \
        --collisionRepolarisation {parameter_dictionary["collisionRepolarisation"]} \
        --repolarisationRate {parameter_dictionary["repolarisationRate"]} \
        --polarityNoiseSigma {parameter_dictionary["polarityNoiseSigma"]}"

    os.system(run_command)

    return None


if __name__ == "__main__":
    main()
