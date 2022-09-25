// Neural.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


/*
  O

O O O

O O O

  O
*/

#include <iostream>
#include <vector>
#include <time.h>

// Calculate the maximum number of connections a neuron can have going into the "dendrites"
int connectionThreshold(float givingCount, float connectionOutward, float receivingCount)
{
    return round((connectionOutward * givingCount) / receivingCount);
}

// Calculate the cost of any observed value relative to it's desired value
float calculateCost(float observed, float desired)
{
    float err = observed - desired;
    return err * err;
}

// Return a random number from 0-1, later being changed to other ranges using the linear
// interpolation function a + (b - a) * t

float RandFloat()
{
    float r = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);
    return r;
}

// Stores basic values of a connection like what the weight is, and who it's connecting

struct Connection
{
public:
    // Neuron giving signal
    int from;
    // Weight of connection
    float weight;
    // Neuron receiving signal
    int to;

    // Simply instantiates values
    Connection(int from, int to, float weight)
    {
        this->from = from;
        this->to = to;
        this->weight = weight;
    }
};


// All logic for a layer, be it input, hidden, or output.

class Layer
{
public:
    // How many neurons in layer
    int layerSize;
    // Which layer it is, input would be 0, first hidden 1...
    int layerIndex;
    // Activation value of each neuron
    std::vector<float> neurons;
    // List of all connections in layer, these are the incoming from previous layer
    std::vector<Connection> conn;
    // Bias of each individual neuron
    std::vector<float> biases;

    // Instantiate values
    Layer(int layerSize, int layerIndex, int bias)
    {
        this->layerSize = layerSize;
        this->layerIndex = layerIndex;

        // Instantiate all neurons and biases as 0 or random
        InstantiateBlankNeuronsAndBiases();
    }

    void InstantiateBlankNeuronsAndBiases()
    {
        // Get current amount of neurons and biases
        int size = neurons.size();
        int bSize = biases.size();

        // Loop through all neurons in layer
        for (int n = 0; n < layerSize; n++)
        {
            // If it is missing a neuron, add it
            if (n >= size)
            {
                // Add it
                neurons.push_back(0);
            }
            else {
                // Reset it
                neurons[n] = 0;
            }

            // If it is missing a bias, add it
            if (n >= bSize)
            {
                // Add it
                biases.push_back(0);
            }
            else {
                // Reset it
                biases[n] = 0;
            }
        }
    }
};

// The Neural Network logic
class Network
{
public:
    // List of all layers
    std::vector<Layer> layers;
    // The unique index of the output layer
    int outputIndex;

    // The gradient of each individual weight, 2 dimensional
    std::vector<std::vector<float>> gradientW;
    // The gradient of each individual bias, 2 dimensional
    std::vector<std::vector<float>> gradientB;

    // Instantiate entire network
    // inputCount: The number of input neurons
    // hiddenLayerCount: The number of hidden layers
    // hiddenLayerSize: The number of neurons in each hiddenLayer
    // outputCount: The number of output neurons
    Network(int inputCount, int hiddenLayerCount, int hiddenLayerSize, int outputCount)
    {
        // Calculate index of the output layer
        outputIndex = hiddenLayerCount + 1;

        // Create input layer
        Layer inputLayer(inputCount, 0, 0);

        // Add it to layers
        layers.push_back(inputLayer);

        // Loop through number of hidden layers
        for (int n = 0; n < hiddenLayerCount; n++)
        {
            // Create new layer
            Layer hiddenLayer(hiddenLayerSize, n + 1, 0);

            // Add it to layers
            layers.push_back(hiddenLayer);
        }

        Layer outputLayer(outputCount, hiddenLayerCount + 1, 0);

        layers.push_back(outputLayer);

        // Loop through all layers skipping the input layer since it has no incoming connections
        for (int x = 1; x < layers.size(); x++)
        {
            // Create second dimension list for both weights and biases
            std::vector<float> innerW;
            std::vector<float> innerB;

            int y;

            for (y = 0; y < layers[x].conn.size(); y++)
            {
                // Add empty weight gradient for each connection in layer
                innerW.push_back(0);
            }

            for (y = 0; y < layers[x].layerSize; y++)
            {
                // Add empty bias gradient for each neuron in layer
                innerB.push_back(0);
            }

            // Add the second dimension to the first
            gradientW.push_back(innerW);
            gradientB.push_back(innerB);
        }
    }

    // This is the logic for connecting ALL layers
    void ConnectAllLayers(int inputToHidden, int hiddenToHidden, int hiddenToOutput)
    {
        // Calculate the max amount of incoming connections for the hidden layers receiving from inputs
        // the hidden layer receving from other hidden layers
        // and the output layer receving from the last hidden layer
        int ith_threshold = connectionThreshold(layers[0].neurons.size(), inputToHidden, layers[1].neurons.size());
        int hth_threshold = connectionThreshold(layers[1].neurons.size(), hiddenToHidden, layers[1].neurons.size());
        int hto_threshold = connectionThreshold(layers[1].neurons.size(), hiddenToOutput, layers[outputIndex].neurons.size());

        // Loop through all layers excluding the input layer as it has no previous layers
        for (int j = 1; j < layers.size(); j++)
        {
            if (j == 1) // If it is the first hidden layer
            {
                // Properly apply logic for connecting input layer and hidden layer
                ConnectLayers(&layers[j - 1], &layers[j], inputToHidden, ith_threshold);
            }
            else if (j == outputIndex) // If it is the last hidden layer connecting to output layer
            {
                // Properly apply logic for connecting last hidden layer and output layer
                ConnectLayers(&layers[j - 1], &layers[j], hiddenToOutput, hto_threshold);
            }
            else { // If it is just a hidden layer connecting to another hidden layer
                // Properly apply logic for connetcion hidden layer to another hidden layer
                ConnectLayers(&layers[j - 1], &layers[j], hiddenToHidden, hth_threshold);
            }
        }
    }

    // This is the logic for connecting two individual layers
    void ConnectLayers(Layer* from, Layer* to, int connectionsOutward, int threshold)
    {
        // How many times another neuron has been connected to, used to prevent neuron connection stacking
        std::vector<int> connCount;
        int n;

        // Set list for every neuron in receving layer, set as 0
        for (n = 0; n < to->neurons.size(); n++)
        {
            connCount.push_back(0);
        }

        // Loop through all giving neurons
        for (n = 0; n < from->neurons.size(); n++)
        {
            // Keep track of all neurons that this neuron has connected to
            std::vector<int> alreadyConnectedNeurons;

            // Loop through amount of connections it is giving out
            for (int c = 0; c < connectionsOutward; c++)
            {
                // These values are used to find the node that has been connected to the least amount of times
                // for an even distribution of connections
                int minIndex = 0;
                int minValue = threshold + 1;

                int t = 0;

                // Loop through all neurons
                for (t = 0; t < to->neurons.size(); t++)
                {
                    // If we have already connected to it, skip
                    if (std::find(alreadyConnectedNeurons.begin(), alreadyConnectedNeurons.end(), t) != alreadyConnectedNeurons.end())
                    {
                        continue;
                    }

                    // If it is our new smallest value
                    if (connCount[t] < minValue)
                    {
                        // Set values accordingly
                        minValue = connCount[t];
                        minIndex = t;
                    }
                }

                // If no more neurons are suitable to connect too, then start connecting other neurons
                if (minValue >= threshold)
                {
                    goto outer;
                }

                // Enumerate
                connCount[minIndex] = connCount[minIndex] + 1;

                // Get random weight
                float randomFloat = RandFloat(); // make it from -25, to 25

                // Make connection
                Connection connection(n, minIndex, randomFloat);

                // Add connection
                from->conn.push_back(connection);

                // Enumerate
                alreadyConnectedNeurons.push_back(minIndex);
            }

        outer:
            continue;
        }
    }

    // Forward Propagation
    std::vector<float> GoForward(std::vector<float> inputs)
    {
        // Set the neuron values of input layers to the inputs parameter
        layers[0].neurons = inputs;

        // Loop through all layers skipping the input layer because it has no previous layers
        for (int j = 1; j < layers.size(); j++)
        {
            int n;

            // Looping through all neurons to reset them
            for (n = 0; n < layers[j].layerSize; n++)
            {
                // Setting neuron as bias, because we are adding later, it is pretty much just adding the bias
                layers[j].neurons[n] = layers[j].biases[n];
            }

            // Loop through all incoming connections
            for (n = 0; n < layers[j - 1].conn.size(); n++)
            {
                // Find the activation of the previous layers neuron
                float power = layers[j - 1].neurons[layers[j - 1].conn[n].from];
                // Find the weight of the connection
                float weight = layers[j - 1].conn[n].weight;
                // Add the two values multiplied to the receving neurons activation
                layers[j].neurons[layers[j - 1].conn[n].to] += power * weight;
            }
        }

        return layers[outputIndex].neurons;
    }

    float Cost(std::vector<float> desired)
    {
        float cost = 0;

        // Summate costs of the output and the desired output
        for (int j = 0; j < layers[outputIndex].layerSize; j++)
        {
            cost += calculateCost(layers[outputIndex].neurons[j], desired[j]);
        }

        return cost;
    }
};

int main()
{
    srand(time(0) * time(0) * time(0));

    //Network(int inputCount, int hiddenLayerCount, int hiddenLayerSize, int outputCount)

    Network network(2, 1, 4, 1);

    //ConnectAllLayers(int inputToHidden, int hiddenToHidden, int hiddenToOutput)
    network.ConnectAllLayers(4, 4, 1);

}
