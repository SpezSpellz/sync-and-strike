using System;
using System.Text;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Phase-0 sanity harness for the on-device learner's optimiser, backprop and trainer.
///
/// Runs five independent checks through the REAL production code path (<see cref="NeuralNetwork"/>,
/// <see cref="PPOTrainer"/>), so a pass means the learner is sound and a failure localises the fault:
///
///   1. Gradient check  - hand-written backprop vs central finite differences of the scalar whose
///                        gradient Backprop is supposed to produce.
///   2. Value descent   - the value head moves toward a regression target.
///   3. Entropy ascent  - a positive entropy coefficient raises entropy.
///   4. Policy sign     - a controlled batch (value forced to 0, unique terminal episodes) where every
///                        arm-1 advantage is +1 and every arm-0 advantage is -1 must RAISE P(arm 1).
///                        This is the check that catches the shuffle-index and sign bugs.
///   5. XOR fit         - the value head + Adam must fit XOR. If it cannot fit 4 rows, the optimiser
///                        is broken and no reward/curriculum tuning can help.
///
/// Run from Tools > Training, or headless:
///   Unity.exe -batchmode -nographics -quit -projectPath &lt;repo&gt; \
///     -executeMethod AdamXorSanityCheck.RunBatch -logFile adam_xor.log
/// </summary>
public static class AdamXorSanityCheck
{
    [MenuItem("Tools/Training/Run Adam XOR Sanity Check")]
    public static void RunFromMenu()
    {
        bool ok = AllPass(out string report);
        Debug.Log(report);
        if (ok) Debug.Log("[AdamXor] ALL CHECKS PASSED");
        else Debug.LogError("[AdamXor] ONE OR MORE CHECKS FAILED");
    }

    /// <summary>Entry point for <c>-executeMethod</c>. Exits non-zero so a batch/CI caller can gate on it.</summary>
    public static void RunBatch()
    {
        bool ok = AllPass(out string report);
        Debug.Log(report);
        if (ok)
        {
            Debug.Log("[AdamXor] ALL CHECKS PASSED");
        }
        else
        {
            Debug.LogError("[AdamXor] ONE OR MORE CHECKS FAILED");
            EditorApplication.Exit(1);
        }
    }

    private static bool AllPass(out string report)
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== Adam / backprop / trainer sanity ===");

        bool g = CheckBackpropGradients(out string gd);
        sb.AppendLine($"[1] backprop gradient check  : {(g ? "PASS" : "FAIL")}");
        sb.AppendLine(gd);

        bool v = CheckValueDescent(out string vd);
        sb.AppendLine($"[2] value descends to target : {(v ? "PASS" : "FAIL")}");
        sb.AppendLine(vd);

        bool e = CheckEntropyAscent(out string ed);
        sb.AppendLine($"[3] entropy increases        : {(e ? "PASS" : "FAIL")}");
        sb.AppendLine(ed);

        bool s = CheckTrainerSignControlled(out string sd);
        sb.AppendLine($"[4] policy sign (controlled) : {(s ? "PASS" : "FAIL")}");
        sb.AppendLine(sd);

        bool x = CheckXorFit(out string xd);
        sb.AppendLine($"[5] XOR fit via Adam         : {(x ? "PASS" : "FAIL")}");
        sb.AppendLine(xd);

        bool all = g && v && e && s && x;
        sb.AppendLine(all ? "=== ALL CHECKS PASSED ===" : "=== ONE OR MORE CHECKS FAILED ===");
        report = sb.ToString();
        return all;
    }

    // 1. Gradient check ---------------------------------------------------
    private static bool CheckBackpropGradients(out string detail)
    {
        var rng = new System.Random(1234567);
        const int input = 3, hidden = 5, actions = 3, scalars = 2, vh = 6;
        var net = new NeuralNetwork(input, hidden, actions, 999, scalars, vh);

        var x = new float[input];
        for (int i = 0; i < input; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        const int action = 1;
        const float dLogProb = 0.6f, dValue = -0.9f, entCoef = 0.15f;
        var dMu = new float[scalars];
        var dLogSd = new float[scalars];
        for (int ss = 0; ss < scalars; ss++)
        {
            dMu[ss] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            dLogSd[ss] = (float)(rng.NextDouble() * 2 - 1) * 0.2f;
        }

        var baseW = net.Export();
        net.ClearGradients();
        net.Forward(x);
        net.Backprop(x, action, dLogProb, dValue, entCoef, dMu, dLogSd);
        net.ApplyGradients(1f, 0f, 1, false);
        var afterW = net.Export();
        net.Import(baseW);

        // L = dLogProb*logProb(a) + dValue*V - entCoef*H + sum dMu*mu + sum dLogSd*logSd
        float Loss(float[] inp)
        {
            net.Forward(inp);
            float lp = NeuralNetwork.LogSoftmax(net.Logits, action);
            float h = Entropy(net.Probabilities);
            float l = dLogProb * lp + dValue * net.Value - entCoef * h;
            var means = net.ScalarMeans;
            var logSds = net.ScalarLogSd;
            for (int ss = 0; ss < scalars; ss++) l += dMu[ss] * means[ss] + dLogSd[ss] * logSds[ss];
            return l;
        }

        const float eps = 1e-3f;
        float worstExcess = 0f;
        string worstName = "";
        int checkedCount = 0;
        var lines = new StringBuilder();

        void CheckTensor(string name, float[] analytic, Action<NeuralNetwork.Weights, int, float> apply)
        {
            int samples = Mathf.Min(8, analytic.Length);
            for (int k = 0; k < samples; k++)
            {
                int i = (int)(rng.NextDouble() * analytic.Length);
                var wp = Clone(baseW); apply(wp, i, +eps); net.Import(wp); float lp = Loss(x);
                var wm = Clone(baseW); apply(wm, i, -eps); net.Import(wm); float lm = Loss(x);
                net.Import(baseW);
                float numeric = (lp - lm) / (2f * eps);
                float ana = analytic[i];
                float tol = 1e-3f + 1e-2f * Mathf.Abs(ana);
                float excess = Mathf.Abs(numeric - ana) / tol;
                if (excess > worstExcess) { worstExcess = excess; worstName = $"{name}[{i}]"; }
                if (excess > 1f) lines.AppendLine($"    {name}[{i}] analytic={ana:0.######} numeric={numeric:0.######}");
                checkedCount++;
            }
        }

        CheckTensor("w1", Sub(baseW.w1, afterW.w1), (w, i, d) => w.w1[i] += d);
        CheckTensor("b1", Sub(baseW.b1, afterW.b1), (w, i, d) => w.b1[i] += d);
        CheckTensor("w2", Sub(baseW.w2, afterW.w2), (w, i, d) => w.w2[i] += d);
        CheckTensor("b2", Sub(baseW.b2, afterW.b2), (w, i, d) => w.b2[i] += d);
        CheckTensor("wp", Sub(baseW.wp, afterW.wp), (w, i, d) => w.wp[i] += d);
        CheckTensor("bp", Sub(baseW.bp, afterW.bp), (w, i, d) => w.bp[i] += d);
        CheckTensor("wv", Sub(baseW.wv, afterW.wv), (w, i, d) => w.wv[i] += d);
        CheckTensor("bv", Sub(baseW.bv, afterW.bv), (w, i, d) => w.bv[i] += d);
        CheckTensor("vw1", Sub(baseW.vw1, afterW.vw1), (w, i, d) => w.vw1[i] += d);
        CheckTensor("vb1", Sub(baseW.vb1, afterW.vb1), (w, i, d) => w.vb1[i] += d);
        CheckTensor("vw2", Sub(baseW.vw2, afterW.vw2), (w, i, d) => w.vw2[i] += d);
        CheckTensor("vb2", Sub(baseW.vb2, afterW.vb2), (w, i, d) => w.vb2[i] += d);
        CheckTensor("wmu", Sub(baseW.wmu, afterW.wmu), (w, i, d) => w.wmu[i] += d);
        CheckTensor("bmu", Sub(baseW.bmu, afterW.bmu), (w, i, d) => w.bmu[i] += d);
        CheckTensor("logSd", Sub(baseW.logSd, afterW.logSd), (w, i, d) => w.logSd[i] += d);

        bool pass = worstExcess <= 1f;
        var sb = new StringBuilder();
        sb.AppendLine($"    {checkedCount} parameters; worst tolerance-excess {worstExcess:0.####} at {worstName}");
        sb.Append(lines);
        detail = sb.ToString();
        return pass;
    }

    // 2. Value descent ----------------------------------------------------
    private static bool CheckValueDescent(out string detail)
    {
        var net = new NeuralNetwork(2, 8, 2, 11, 0, 8);
        var x = new float[] { 0.4f, 0.6f };
        const float target = 1.0f;
        net.Forward(x); float before = net.Value;
        net.ClearGradients();
        net.Forward(x);
        net.Backprop(x, 0, 0f, net.Value - target, 0f);
        net.ApplyGradients(0.5f, 0f, 1, false);
        net.Forward(x); float after = net.Value;
        bool pass = Mathf.Abs(after - target) < Mathf.Abs(before - target) + 1e-6f;
        detail = $"    V before={before:0.####} after={after:0.####} target={target} (should approach)";
        return pass;
    }

    // 3. Entropy ascent ---------------------------------------------------
    private static bool CheckEntropyAscent(out string detail)
    {
        var net = new NeuralNetwork(2, 8, 4, 12, 0, 8);
        var x = new float[] { 0.2f, -0.3f };
        var w = net.Export();
        w.bp[0] += 3f;
        net.Import(w);
        net.Forward(x); float before = Entropy(net.Probabilities);
        net.ClearGradients();
        net.Forward(x);
        net.Backprop(x, 0, 0f, 0f, 0.5f);
        net.ApplyGradients(0.5f, 0f, 1, false);
        net.Forward(x); float after = Entropy(net.Probabilities);
        bool pass = after > before + 1e-4f;
        detail = $"    H before={before:0.####} after={after:0.####} (should rise)";
        return pass;
    }

    // 4. Controlled policy-sign test --------------------------------------
    private static bool CheckTrainerSignControlled(out string detail)
    {
        var net = new NeuralNetwork(2, 8, 2, 777, 0, 8);
        var hyper = new PPOTrainer.Hyper
        {
            learningRate = 0.1f, gamma = 0.99f, lambda = 0.95f, valueLambda = 1f,
            clipEpsilon = 0.2f, valueCoef = 0f, entropyCoef = 0f, epochs = 1,
            minibatch = 8, targetKl = 0f, useAdam = true, batchSize = 8, bufferSize = 16,
            maxGradNorm = 0f,
        };
        var trainer = new PPOTrainer(net, hyper, new System.Random(99));
        var obs = new float[] { 0.5f, -0.2f };
        int[] actions = { 1, 0, 1, 0, 1, 0, 1, 0 };
        float[] rewards = { 1, -1, 1, -1, 1, -1, 1, -1 };
        for (int i = 0; i < 8; i++)
        {
            net.Forward(obs);
            float lp = NeuralNetwork.LogSoftmax(net.Logits, actions[i]);
            // value forced to 0 + unique episode -> advantage is exactly the reward (+1/-1), no GAE,
            // and no adaptive sampling to confound the direction.
            trainer.Add(obs, obs, actions[i], lp, 0f, rewards[i], true, null, 0, i);
        }
        net.Forward(obs);
        float before = net.Probabilities[1];
        trainer.Update();
        net.Forward(obs);
        float after = net.Probabilities[1];
        bool pass = after > before + 1e-4f;
        detail = $"    P(arm1) before={before:0.####} after={after:0.####} "
               + $"(advantages exactly +1/-1, correct PPO must raise P(arm1))";
        return pass;
    }

    // 5. XOR fit ----------------------------------------------------------
    private static readonly float[][] XorIn =
    {
        new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f },
    };
    private static readonly float[] XorOut = { 0f, 1f, 1f, 0f };

    private static bool CheckXorFit(out string detail)
    {
        const int hidden = 16;
        var net = new NeuralNetwork(2, hidden, 2, 20250101, 0, hidden);
        const float lr = 0.02f;
        const int iters = 4000;
        var rng = new System.Random(7);
        for (int it = 0; it < iters; it++)
        {
            int i = rng.Next(XorIn.Length);
            net.ClearGradients();
            net.Forward(XorIn[i]);
            net.Backprop(XorIn[i], 0, 0f, net.Value - XorOut[i], 0f);
            net.ApplyGradients(lr, 0f, 1, true);
        }
        float mse = 0f;
        var sb = new StringBuilder();
        for (int i = 0; i < XorIn.Length; i++)
        {
            net.Forward(XorIn[i]);
            float err = net.Value - XorOut[i];
            mse += err * err;
            sb.AppendLine($"    ({XorIn[i][0]},{XorIn[i][1]}) -> {net.Value:0.####} (target {XorOut[i]:0.##})");
        }
        mse /= XorIn.Length;
        sb.AppendLine($"    final MSE {mse:0.######} after {iters} Adam steps (lr {lr})");
        detail = sb.ToString();
        return mse < 0.01f;
    }

    // --------------------------------------------------------------------
    private static float Entropy(float[] p)
    {
        float h = 0f;
        for (int i = 0; i < p.Length; i++) if (p[i] > 0f) h -= p[i] * Mathf.Log(p[i]);
        return h;
    }

    private static float[] Sub(float[] a, float[] b)
    {
        var r = new float[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = a[i] - b[i];
        return r;
    }

    private static NeuralNetwork.Weights Clone(NeuralNetwork.Weights w)
    {
        return new NeuralNetwork.Weights
        {
            input = w.input, hidden = w.hidden, actions = w.actions, valueHidden = w.valueHidden,
            w1 = (float[])w.w1.Clone(), b1 = (float[])w.b1.Clone(),
            w2 = (float[])w.w2.Clone(), b2 = (float[])w.b2.Clone(),
            wp = (float[])w.wp.Clone(), bp = (float[])w.bp.Clone(),
            wv = (float[])w.wv.Clone(), bv = (float[])w.bv.Clone(),
            wmu = w.wmu != null ? (float[])w.wmu.Clone() : null,
            bmu = w.bmu != null ? (float[])w.bmu.Clone() : null,
            logSd = w.logSd != null ? (float[])w.logSd.Clone() : null,
            vw1 = (float[])w.vw1.Clone(), vb1 = (float[])w.vb1.Clone(),
            vw2 = (float[])w.vw2.Clone(), vb2 = (float[])w.vb2.Clone(),
        };
    }
}
