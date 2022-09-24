// Neural.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>

int connectionThreshold(float givingCount, float connectionOutward, float receivingCount)
{
    return round((connectionOutward * givingCount) / receivingCount);
}

float RandFloat()
{
    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    return r;
}

struct Connection
{
public:
    int from;
    float weight;
    int to;

    Connection(int from, int to, float weight)
    {
        this->from = from;
        this->to = to;
        this->weight = weight;
    }
};

class Layer
{
public:
    int layerSize;
    int layerIndex;
    std::vector<float> neurons;
    std::vector<Connection> conn;

    Layer(int layerSize, int layerIndex)
    {
        this->layerSize = layerSize;
        this->layerIndex = layerIndex;

        InstantiateBlankNeurons();
    }
    
    int bias;

    void InstantiateBlankNeurons()
    {
        int size = neurons.size();

        for (int n = 0; n < layerSize; n++)
        {
            if (n >= size)
            {
                neurons.push_back(0);
            }
            else {
                neurons[n] = 0;
            }
        }
    }
};

class Network
{
public:
    std::vector<Layer> layers;
    int outputIndex;

    Network(int inputCount, int hiddenLayerCount, int hiddenLayerSize, int outputCount)
    {
        outputIndex = hiddenLayerCount + 2;

        Layer inputLayer(inputCount, 0);

        layers.push_back(inputLayer);

        for (int n = 0; n < hiddenLayerCount; n++)
        {
            Layer hiddenLayer(hiddenLayerSize, n + 1);

            layers.push_back(hiddenLayer);
        }

        Layer outputLayer(outputCount, hiddenLayerCount + 2);

        layers.push_back(outputLayer);
    }

    void ConnectAllLayers(int inputToHidden, int hiddenToHidden, int hiddenToOutput)
    {
        int ith_threshold = connectionThreshold(layers[0].neurons.size(), inputToHidden, layers[1].neurons.size());
        int hth_threshold = connectionThreshold(layers[1].neurons.size(), hiddenToHidden, layers[1].neurons.size());
        int hto_threshold = connectionThreshold(layers[1].neurons.size(), hiddenToOutput, layers[outputIndex-1].neurons.size());

        for (int j = 1; j < layers.size(); j++)
        {
            if (j == 1)
            {
                ConnectLayers(&layers[j - 1], &layers[j], inputToHidden, ith_threshold);
            }
            else if (j == outputIndex-1)
            {
                ConnectLayers(&layers[j - 1], &layers[j], hiddenToOutput, hto_threshold);
            }
            else {
                ConnectLayers(&layers[j - 1], &layers[j], hiddenToHidden, hth_threshold);
            }
        }
    }

    void ConnectLayers(Layer* from, Layer* to, int connectionsOutward, int threshold)
    {
        std::vector<int> connCount;
        int n;

        for (n = 0; n < to->neurons.size(); n++)
        {
            connCount.push_back(0);
        }

        for (n = 0; n < from->neurons.size(); n++)
        {
            for (int c = 0; c < connectionsOutward; c++)
            {
                int t = 0;

                for (t = 0; t < to->neurons.size(); t++)
                {
                    if (connCount[t] < threshold)
                    {
                        break;
                    }
                }

                float randomFloat = RandFloat();
                randomFloat = 0.5;

                Connection connection(n, t, randomFloat);
                from->conn.push_back(connection);
            }
        }
    }

    std::vector<float> GoForward(std::vector<float> inputs)
    {
        layers[0].neurons = inputs;

        for (int j = 1; j < layers.size(); j++)
        {
            int n;

            for (n = 0; n < layers[j].neurons.size(); n++)
            {
                layers[j].neurons[n] = 0;
            }

            for (n = 0; n < layers[j - 1].conn.size(); n++)
            {
                Connection* conn = &layers[j - 1].conn[n];
                float power = layers[j - 1].neurons[conn->from];
                float weight = conn->weight;
                layers[j].neurons[conn->to] += power * weight;
            }
        }

        return layers[outputIndex - 1].neurons;
    }
};

int main()
{
    Network network(2, 1, 2, 1);
    network.ConnectAllLayers(2, 1, 1);

    std::vector<float> inputs;
    inputs.push_back(1);
    inputs.push_back(0.5F);

    std::vector<float> output = network.GoForward(inputs);
    std::cout << network.layers[2].conn.size() << std::endl;
    std::cout << output[0] << std::endl;
}
