using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A small clipped-surrogate PPO implementation for the companion's on-device policy.
///
/// One environment step is one turn. Transitions are (observation, chosen action, log-prob, value,
/// reward, done). The trainer buffers turns, computes GAE advantages, and runs a few epochs of
/// minibatch updates against the shared <see cref="NeuralNetwork"/>.
///
/// This deliberately has no external dependency (no ML-Agents, no Python) so the shipped build can
/// keep training the companion on the player's machine. The enemy stays frozen and never uses this.
/// </summary>
public class PPOTrainer
{
    public class Transition
    {
        public float[] obs;
        public float[] nextObs;
        public int action;
        public float logProb;   // log-prob at collection time (the PPO ratio denominator)
        public float value;     // value estimate at collection time
        public float reward;
        public bool done;
    }

    [Serializable]
    public class Hyper
    {
        public float learningRate = 3e-4f;
        public float gamma = 0.99f;
        public float lambda = 0.95f;
        public float clipEpsilon = 0.2f;
        public float valueCoef = 0.5f;
        public float entropyCoef = 0.01f;
        public int epochs = 4;
        public int minibatch = 32;
        public int bufferSize = 512;
    }

    private readonly NeuralNetwork net;
    private readonly Hyper hyper;
    private readonly List<Transition> buffer = new List<Transition>();
    private readonly System.Random rng;
    private int[] shuffle;

    public PPOTrainer(NeuralNetwork net, Hyper hyper, System.Random rng)
    {
        this.net = net;
        this.hyper = hyper;
        this.rng = rng;
        shuffle = new int[hyper.bufferSize];
    }

    public int Buffered => buffer.Count;
    public int UpdateCount { get; private set; }

    public void Add(float[] obs, float[] nextObs, int action, float logProb, float value, float reward, bool done)
    {
        buffer.Add(new Transition
        {
            obs = obs, nextObs = nextObs, action = action,
            logProb = logProb, value = value, reward = reward, done = done,
        });
        if (buffer.Count > hyper.bufferSize) buffer.RemoveAt(0);
    }

    /// <summary>Apply a late reward adjustment (e.g. the player's vote) to the most recent turn.</summary>
    public void AdjustLastReward(float delta)
    {
        if (buffer.Count > 0) buffer[buffer.Count - 1].reward += delta;
    }

    public bool ReadyToUpdate() => buffer.Count >= hyper.minibatch;

    /// <summary>Run the PPO update over the current buffer. Consumes nothing; buffer persists until trimmed.</summary>
    public void Update()
    {
        int n = buffer.Count;
        if (n < 2) return;

        // --- GAE advantages ---
        var advantage = new float[n];
        var returns = new float[n];
        float gae = 0f;
        for (int t = n - 1; t >= 0; t--)
        {
            var tr = buffer[t];
            float nextValue = 0f;
            if (!tr.done) nextValue = net.Forward(tr.nextObs);
            float delta = tr.reward + hyper.gamma * nextValue - tr.value;
            gae = delta + (tr.done ? 0f : hyper.gamma * hyper.lambda * gae);
            advantage[t] = gae;
            returns[t] = gae + tr.value;
        }

        // Normalise advantages for a stable step size.
        float mean = 0f;
        for (int t = 0; t < n; t++) mean += advantage[t];
        mean /= n;
        float varSum = 0f;
        for (int t = 0; t < n; t++) { float d = advantage[t] - mean; varSum += d * d; }
        float std = Mathf.Sqrt(varSum / n) + 1e-8f;
        for (int t = 0; t < n; t++) advantage[t] = (advantage[t] - mean) / std;

        // --- Clipped PPO epochs ---
        for (int epoch = 0; epoch < hyper.epochs; epoch++)
        {
            Shuffle(n);
            for (int start = 0; start < n; start += hyper.minibatch)
            {
                int end = Mathf.Min(start + hyper.minibatch, n);
                int count = end - start;
                if (count == 0) continue;
                net.ClearGradients();

                for (int i = start; i < end; i++)
                {
                    var tr = buffer[shuffle[i]];
                    float value = net.Forward(tr.obs);
                    float newLogProb = LogProb(tr.obs, tr.action);
                    float ratio = Mathf.Exp(newLogProb - tr.logProb);

                    // Textbook clipping mask: when the clipped branch is the active (smaller) term,
                    // the ratio is pinned and contributes no policy gradient.
                    float A = advantage[i];
                    bool useClipped = (A > 0f && ratio > 1f + hyper.clipEpsilon)
                                    || (A < 0f && ratio < 1f - hyper.clipEpsilon);
                    float dLogProb = useClipped ? 0f : A * ratio;
                    float dValue = hyper.valueCoef * (value - returns[i]);

                    net.Backprop(tr.obs, tr.action, dLogProb, dValue, hyper.entropyCoef);
                }
                net.ApplyGradients(hyper.learningRate / count);
            }
        }
        UpdateCount++;
    }

    private float LogProb(float[] obs, int action)
    {
        net.Forward(obs);
        return NeuralNetwork.LogSoftmax(net.Logits, action);
    }

    private void Shuffle(int n)
    {
        if (shuffle.Length < n)
            shuffle = new int[Mathf.NextPowerOfTwo(Mathf.Max(n, 16))];
        for (int i = 0; i < n; i++) shuffle[i] = i;
        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            int tmp = shuffle[i]; shuffle[i] = shuffle[j]; shuffle[j] = tmp;
        }
    }
}