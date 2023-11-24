#include "agents.h"

#include <algorithm>
#include <random>
#include <cmath>
#include <cassert>
#include <limits>
#include <iostream>

#include <boost/numeric/ublas/matrix.hpp>
namespace boostMatrix = boost::numeric::ublas;
using std::atan2;

typedef std::vector<std::tuple<int, int>> vectorOfTuples;

vectorOfTuples HEADING_CONTACT_MAPPING{
    std::tuple<int, int>{1, 0},
    std::tuple<int, int>{0, 0},
    std::tuple<int, int>{0, 1},
    std::tuple<int, int>{0, 2},
    std::tuple<int, int>{1, 2},
    std::tuple<int, int>{2, 2},
    std::tuple<int, int>{2, 1},
    std::tuple<int, int>{2, 0},
    std::tuple<int, int>{1, 0}
};

// Constructor:
CellAgent::CellAgent(
    double startX, double startY, double startHeading,
    unsigned int setCellSeed, int setCellID, int setCellType,
    double setWbK, double setKappa,
    double setHomotypicInhibition, double setHeterotypicInhibition,
    double setPolarityPersistence, double setPolarityTurningCoupling,
    double setFlowScaling, double setFlowPolarityCoupling
    )
    : thereIsMatrixInteraction{true}
    , x{startX}
    , y{startY}
    , polarityX{0.001 * cos(startHeading)}
    , polarityY{0.001 * sin(startHeading)}
    , actinFlow{0}
    , flowPolarityCoupling{setFlowPolarityCoupling}
    , flowScaling{setFlowScaling}
    , polarityPersistence{setPolarityPersistence}
    , polarityTurningCoupling{setPolarityTurningCoupling}
    , attachmentHistory{std::vector<double>(0, 0.0)}
    , kappa{setKappa}
    , localMatrixHeading{boostMatrix::zero_matrix<double>(3, 3)}
    , localMatrixPresence{boostMatrix::zero_matrix<bool>(3, 3)}
    , directionalInfluence{M_PI}
    , directionalIntensity{0}
    , cellType0ContactState{boostMatrix::zero_matrix<bool>(3, 3)}
    , cellType1ContactState{boostMatrix::zero_matrix<bool>(3, 3)}
    , localCellHeadingState{boostMatrix::zero_matrix<double>(3, 3)}
    , cellSeed{setCellSeed}
    , cellID{setCellID}
    , cellType{setCellType}
    , kWeibull{setWbK}
    , homotypicInhibitionRate{setHomotypicInhibition}
    , heterotypicInhibitionRate{setHeterotypicInhibition}
    , sampledAngle{0}
{
    // Initialising randomness:
    seedGenerator = std::mt19937(cellSeed);
    seedDistribution = std::uniform_int_distribution<unsigned int>(0, UINT32_MAX);

    // Initialising influence selector:
    generatorInfluence = std::mt19937(seedDistribution(seedGenerator));

    // Initialising von Mises distribution:
    generatorU1 = std::mt19937(seedDistribution(seedGenerator));
    generatorU2 = std::mt19937(seedDistribution(seedGenerator));
    generatorB = std::mt19937(seedDistribution(seedGenerator));

    uniformDistribution = std::uniform_real_distribution<double>(0, 1);
    bernoulliDistribution = std::bernoulli_distribution();

    // Initialising Weibull distribution:
    generatorWeibull = std::mt19937(seedDistribution(seedGenerator));

    // Generators and distribution for contact inhibition calculations:
    generatorAngleUniform = std::mt19937(seedDistribution(seedGenerator));
    generatorCornerCorrection = std::mt19937(seedDistribution(seedGenerator));
    generatorForInhibitionRate = std::mt19937(seedDistribution(seedGenerator));
    angleUniformDistribution = std::uniform_real_distribution<double>(-M_PI, M_PI);
}
// Public Definitions:

// Getters:
// Getters for values that shouldn't change:
double CellAgent::getID() {return cellID;}
double CellAgent::getHomotypicInhibitionRate() {return homotypicInhibitionRate;}
double CellAgent::getCellType() {return cellType;}

// Persistent variable getters: (these variables represent aspects of the cell's "memory")
double CellAgent::getX() {return x;}
double CellAgent::getY() {return y;}
std::tuple<double, double> CellAgent::getPosition() {return std::tuple<double, double>{x, y};}
double CellAgent::getPolarity() {return findPolarityDirection();}
double CellAgent::getPolarityExtent() {return findPolarityExtent();}
std::vector<double> CellAgent::getAttachmentHistory() {return attachmentHistory;}

// Instantaneous variable getters: (these variables are updated each timestep,
// and represent the cell's percepts)
double CellAgent::getMovementDirection() {return movementDirection;}
double CellAgent::getActinFlow() {return actinFlow;}
double CellAgent::getDirectionalInfluence() {return directionalInfluence;}
double CellAgent::getDirectionalIntensity() {return directionalIntensity;}
double CellAgent::getDirectionalShift() {return directionalShift;}
double CellAgent::getSampledAngle() {return sampledAngle;}


// Setters:
void CellAgent::setPosition(std::tuple<double, double> newPosition) {
    x = std::get<0>(newPosition);
    y = std::get<1>(newPosition);
}

void CellAgent::setContactStatus(boostMatrix::matrix<bool> stateToSet, int cellType) {
    if (cellType == 0) {
        cellType0ContactState = stateToSet;
    } else {
        cellType1ContactState = stateToSet;
    }
}

void CellAgent::setLocalCellHeadingState(boostMatrix::matrix<double> stateToSet) {
    localCellHeadingState = stateToSet;
}

void CellAgent::setLocalMatrixHeading(boostMatrix::matrix<double> stateToSet) {
    localMatrixHeading = stateToSet;
}

void CellAgent::setLocalMatrixPresence(boostMatrix::matrix<bool> stateToSet) {
    localMatrixPresence = stateToSet;
}

// Simulation code:
void CellAgent::takeRandomStep() {
    // // Calculating new attachment:
    if (thereIsMatrixInteraction) {
        // Calculate mean for von mises distribution by taking average of local ECM:
        double relativeECMDirection;
        double ecmCoherence;
        std::tie(relativeECMDirection, ecmCoherence) = getAverageDeltaHeading();

        // Recording values as member variables to be output at end of global timestep:
        directionalInfluence = relativeECMDirection;
        directionalIntensity = ecmCoherence;
    }

    // if (cellID == 1) {
    //     std::cout << relativeECMDirection << " ~~~~ " << ecmCoherence << "\n";
    // }

    // Sample von mises distribution:
    double angleDelta = sampleVonMises(kappa);
    sampledAngle = angleDelta;
    double vonMisesMean{0};
    if (uniformDistribution(generatorInfluence) < directionalIntensity) {
        vonMisesMean = directionalInfluence;
    }
    directionalShift = angleMod(angleDelta + vonMisesMean);

    // Finding angle mean of current polarity and proposed movement direction:
    double proposedMovementDirection = angleMod(findPolarityDirection() + directionalShift);
    double shiftWeighting{1-(findPolarityExtent()*polarityTurningCoupling)};

    double movX{shiftWeighting*cos(proposedMovementDirection) + (1-shiftWeighting)*polarityX};
    double movY{shiftWeighting*sin(proposedMovementDirection) + (1-shiftWeighting)*polarityY};

    movementDirection = atan2(movY, movX);

    // double newAttachmentDirection = angleMod(findPolarityDirection() + angleDelta);
    // attachmentHistory.insert(attachmentHistory.begin(), newAttachmentDirection);
    // while (attachmentHistory.size() > 3) {
    //     attachmentHistory.pop_back();
    // }
    // std::cout << attachmentHistory[0] << "\n";
    // // Calculate movement direction using polarity-weighted average of attachments:
    // movementDirection = getAverageAttachmentHeading();

    double collisionRepolarisation{0.75}; // [-1, 1]
    double repolarisationRate{0.5}; // [0, 1]

    // // Calculating actin flow:
    // Calculate weibull lambda based on absolute value of polarity:
    if (checkForCollisions()) {
        actinFlow = 0;

        double polarityChangeX{collisionRepolarisation*cos(movementDirection)};
        double polarityChangeY{collisionRepolarisation*sin(movementDirection)};

        polarityX = (1-repolarisationRate)*polarityX + repolarisationRate*polarityChangeX;
        polarityY = (1-repolarisationRate)*polarityY + repolarisationRate*polarityChangeY;

        // Return without updating position - collision has occured at this point:    
        return;
    } else {
        double actinModulator{sqrt(pow(polarityX, 2) + pow(polarityY, 2)) * flowScaling};
        std::weibull_distribution<double> weibullDistribution{
            std::weibull_distribution<double>(kWeibull, actinModulator)
        };
        actinFlow = weibullDistribution(generatorWeibull);
    }

    // // Update position:
    double dx{actinFlow * cos(movementDirection)};
    double dy{actinFlow * sin(movementDirection)};
    x = x + dx;
    y = y + dy;

    // // Update polarity based on movement direction and actin flow:
    double polarityChangeExtent{tanh(actinFlow*flowPolarityCoupling)};
    double polarityChangeX{polarityChangeExtent*cos(movementDirection)};
    double polarityChangeY{polarityChangeExtent*sin(movementDirection)};

    double newPolarityX{polarityPersistence*polarityX + (1-polarityPersistence)*polarityChangeX};
    double newPolarityY{polarityPersistence*polarityY + (1-polarityPersistence)*polarityChangeY};

    // Clamping polarity components to [-1, 1] (while preserving direction):
    double newPolarityExtent{sqrt(pow(newPolarityX, 2) + pow(newPolarityY, 2))};
    assert(newPolarityExtent <= 1);
    double direction{atan2(newPolarityY, newPolarityX)};
    polarityX = newPolarityExtent * cos(direction);
    polarityY = newPolarityExtent * sin(direction);
}

// Private functions for cell behaviour:
bool CellAgent::checkForCollisions() {
    // This checks for which cell in the Moore neighbourhood we need to query:
    int stateIndex{int(floor((movementDirection + M_PI + (M_PI/8)) / (M_PI / 4)))};
    int iContact{std::get<0>(HEADING_CONTACT_MAPPING[stateIndex])};
    int jContact{std::get<1>(HEADING_CONTACT_MAPPING[stateIndex])};

    // Are any cells nearby?
    bool type0Contact = cellType0ContactState(iContact, jContact);
    bool type1Contact = cellType1ContactState(iContact, jContact);
    if ((!type0Contact) && (!type1Contact)) {
        return false;
    }

    // Calculate corner collision correction:
    bool isCorner{(stateIndex == 1 || stateIndex == 3 || stateIndex == 5 || stateIndex == 7)};
    const double cornerCorrectionFactor{0.5 + ((9*M_PI)/32) - ((9*sqrt(2))/16)};

    if (isCorner) {
        bool skipCollision{uniformDistribution(generatorCornerCorrection) > cornerCorrectionFactor};
        if (skipCollision) {return false;}
    }

    // Calculate correction for relative cell angles:
    double otherCellHeading{localCellHeadingState(iContact, jContact)};
    double angularDistance{calculateMinimumAngularDistance(findPolarityDirection(), otherCellHeading)};
    double angularCorrectionFactor{pow(cos(angularDistance), 2)};

    // Calculate homotypic / heterotypic interactions:
    std::array<bool, 2> cellTypeContacts{type0Contact, type1Contact};
    if (cellTypeContacts[cellType]) {
        // Do homotypic interaction if present:
        double effectiveHomotypicInhibition{(homotypicInhibitionRate*angularCorrectionFactor)};
        if (uniformDistribution(generatorForInhibitionRate) < effectiveHomotypicInhibition) {
            return true;
        }
    } else if (cellTypeContacts[std::abs(cellType - 1)]) {
        // Do heterotypic interaction if no homotypic interaction:
        double effectiveHeterotypicInhibition{
            (heterotypicInhibitionRate*angularCorrectionFactor)
        };
        if (uniformDistribution(generatorForInhibitionRate) < effectiveHeterotypicInhibition) {
            return true;
        }
    }

    // If no collision is ultimately calculated, return no collision:
    return false;
}

double CellAgent::sampleVonMises(double kappa) {
    // See Efficient Simulation of the von Mises Distribution - Best & Fisher 1979
    // Calculating distribution parameters:
    double vm_tau = 1 + sqrt(1 + (4 * pow(kappa, 2)));
    double vm_rho = (vm_tau - sqrt(2 * vm_tau)) / (2 * kappa);
    double vm_r = (1 + pow(vm_rho, 2)) / (2 * vm_rho);

    // Performing sampling:
    bool angleSampled{false};
    double sampledVonMises;
    while (!angleSampled) {
        // Sample from our distributions:
        double U1{uniformDistribution(generatorU1)};
        double U2{uniformDistribution(generatorU2)};
        int B{int(bernoulliDistribution(generatorB))};

        // Calculate derived values:
        double z{cos(M_PI * U1)};
        double f{(1 + vm_r * z) / (vm_r + z)};
        double c{kappa * (vm_r - f)};

        // Determine whether we accept the result:
        bool simpleCondition{(c*(2 - c) - U2) > 0};
        if (simpleCondition) {
            angleSampled = true;
            int angleSign{(2*B) - 1};
            sampledVonMises = angleSign * acos(f);
            continue;
        }

        bool logCondition{(log(c/U2) + 1 - c) > 0};
        if (logCondition) {
            angleSampled = true;
            int angleSign{(2*B) - 1};
            sampledVonMises = angleSign * acos(f);
        }
    }
    return sampledVonMises;
}

double CellAgent::angleMod(double angle) {
    while (angle < -M_PI) {angle += 2*M_PI;};
    while (angle >= M_PI) {angle -= 2*M_PI;};
    return angle;
}

double CellAgent::calculateMinimumAngularDistance(double headingA, double headingB) {
    // Calculating change in theta:
    double deltaHeading{headingA - headingB};
    while (deltaHeading <= -M_PI) {deltaHeading += 2*M_PI;}
    while (deltaHeading > M_PI) {deltaHeading -= 2*M_PI;}

    double flippedHeading;
    if (deltaHeading < 0) {
        flippedHeading = M_PI + deltaHeading;
    } else {
        flippedHeading = -(M_PI - deltaHeading);
    }

    // Selecting smallest change in theta and ensuring correct range:
    if (std::abs(deltaHeading) < std::abs(flippedHeading)) {
        assert((std::abs(deltaHeading) <= M_PI/2));
        return deltaHeading;
    } else {
        assert((std::abs(flippedHeading) <= M_PI/2));
        return flippedHeading;
    };
}

std::tuple<double, double> CellAgent::getAverageDeltaHeading() {
    std::array<int, 3> rowScan = {0, 1, 2};
    std::array<int, 3> columnScan = {0, 1, 2};
    double polarityDirection{findPolarityDirection()};

    // Getting all headings:
    std::vector<double> deltaHeadingVector;
    for (auto & row : rowScan)
    {
        for (auto & column : columnScan)
        {
            if (localMatrixPresence(row, column)) {
                double ecmHeading{localMatrixHeading(row, column)};
                deltaHeadingVector.push_back(
                    calculateCellDeltaTowardsECM(ecmHeading, polarityDirection)
                );
            }
        }
    }

    // If no ECM present, return angle with no influence:
    if (deltaHeadingVector.empty()) {
        return {0, 0};
    }

    // Getting the angle average:
    double sineMean{0};
    double cosineMean{0};
    for (auto & heading : deltaHeadingVector) {
        sineMean += sin(heading);
        cosineMean += cos(heading);
    }

    sineMean /= 9;
    cosineMean /= 9;

    assert(abs(sineMean) <= 1);
    assert(abs(cosineMean) <= 1);
    double angleAverage{atan2(sineMean, cosineMean)};
    double angleIntensity{sqrt(pow(sineMean, 2) + pow(cosineMean, 2))};
    return {angleAverage, angleIntensity};
}


double CellAgent::calculateCellDeltaTowardsECM(double ecmHeading, double cellHeading) {
    // Ensuring input values are in the correct range:
    assert((ecmHeading >= 0) & (ecmHeading < M_PI));
    assert((cellHeading >= -M_PI) & (cellHeading < M_PI));

    // Calculating change in theta (ECM is direction agnostic so we have to reverse it):
    double deltaHeading{ecmHeading - cellHeading};
    // while (deltaHeading <= -M_PI) {deltaHeading += M_PI;}
    while (deltaHeading > M_PI) {deltaHeading -= M_PI;}

    double flippedHeading;
    if (deltaHeading < 0) {
        flippedHeading = M_PI + deltaHeading;
    } else {
        flippedHeading = -(M_PI - deltaHeading);
    };

    // Selecting smallest change in theta and ensuring correct range:
    if (std::abs(deltaHeading) < std::abs(flippedHeading)) {
        assert((std::abs(deltaHeading) <= M_PI/2));
        return deltaHeading;
    } else {
        assert((std::abs(flippedHeading) <= M_PI/2));
        return flippedHeading;
    };
}

double CellAgent::findPolarityDirection() {
    if ((polarityY == 0) & (polarityX == 0)) {
        assert(false);
        return 0;
    } else {
        return atan2(polarityY, polarityX);
    };
};


double CellAgent::findPolarityExtent() {
    double polarityExtent{sqrt(pow(polarityX, 2) + pow(polarityY, 2))};
    assert(polarityExtent <= 1.0);
    return polarityExtent;
};


double CellAgent::getAverageAttachmentHeading() {
    double sineMean{0};
    double cosineMean{0};
    for (auto & heading : attachmentHistory) {
        sineMean += sin(heading);
        cosineMean += cos(heading);
    }

    sineMean /= 5;
    cosineMean /= 5;

    assert(std::abs(sineMean) <= 1);
    assert(std::abs(cosineMean) <= 1);
    double angleAverage{atan2(sineMean, cosineMean)};
    return angleAverage;
}