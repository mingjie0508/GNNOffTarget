<script lang="ts">
    import * as echarts from "echarts";
    import { onMount } from "svelte";
    import { openGAStreamURL, buildStreamURL, type Frame } from "$lib/stream";

    // --- Form state ---
    let lambda = 0.5;
    let generations = 50;
    let population = 100;
    let targetSequence = "ACGTACGTACGTACGTACGT"; // 20bp default
    let targetSequenceError = "";

    // --- Run state ---
    let running = false;
    let statusText = "Idle";
    let es: EventSource | null = null;

    // --- Chart data ---
    let gens: number[] = [];
    let bestFitData: number[] = [];
    let meanFitData: number[] = [];

    // --- Best sequence visualization ---
    let currentBestSequence = "";
    let previousBestSequence = "";
    let flashingIndices = new Set<number>();

    // --- Sequence logo data ---
    let logoData: Record<
        string,
        { A: number; C: number; G: number; T: number }
    > = {};

    // --- ECharts instances ---
    let chartFitness: echarts.ECharts | null = null;
    let chartLogo: echarts.ECharts | null = null;

    function validateTargetSequence(seq: string): string {
        if (!seq) return "Target sequence is required";
        if (!/^[ACGT]*$/i.test(seq))
            return "Target sequence must contain only A, C, G, T nucleotides";
        if (seq.length !== 20)
            return "Target sequence must be exactly 20 base pairs";
        return "";
    }

    function resetData() {
        gens = [];
        bestFitData = [];
        meanFitData = [];
        currentBestSequence = "";
        previousBestSequence = "";
        flashingIndices.clear();
        logoData = {};
        chartFitness?.setOption({
            xAxis: { data: [] },
            series: [{ data: [] }, { data: [] }],
        });
        chartLogo?.setOption({
            series: [{ data: [] }, { data: [] }, { data: [] }, { data: [] }],
        });
    }

    function initCharts() {
        const el1 = document.getElementById("fitness")!;
        chartFitness = echarts.init(el1);
        chartFitness.setOption({
            title: { text: "Fitness Progress" },
            tooltip: { trigger: "axis" },
            legend: { data: ["Best", "Mean"] },
            xAxis: { type: "category", data: [] },
            yAxis: { type: "value" },
            series: [
                { name: "Best", type: "line", data: [] },
                { name: "Mean", type: "line", data: [] },
            ],
        });

        const el2 = document.getElementById("logo")!;
        chartLogo = echarts.init(el2);
        chartLogo.setOption({
            title: { text: "Sequence Logo - Base Distribution" },
            tooltip: {
                trigger: "axis",
                axisPointer: { type: "shadow" },
                formatter: function (params: any) {
                    let result = `Position ${params[0].axisValue}:<br/>`;
                    params.forEach((param: any) => {
                        result += `${param.seriesName}: ${param.value}<br/>`;
                    });
                    return result;
                },
            },
            legend: { data: ["A", "C", "G", "T"] },
            xAxis: {
                type: "category",
                data: Array.from({ length: 20 }, (_, i) => (i + 1).toString()),
                name: "Position",
            },
            yAxis: {
                type: "value",
                name: "Count",
                min: 0,
            },
            series: [
                {
                    name: "A",
                    type: "bar",
                    stack: "nucleotides",
                    data: [],
                    itemStyle: { color: "#fbbf24" }, // amber
                },
                {
                    name: "C",
                    type: "bar",
                    stack: "nucleotides",
                    data: [],
                    itemStyle: { color: "#ef4444" }, // red
                },
                {
                    name: "G",
                    type: "bar",
                    stack: "nucleotides",
                    data: [],
                    itemStyle: { color: "#10b981" }, // emerald
                },
                {
                    name: "T",
                    type: "bar",
                    stack: "nucleotides",
                    data: [],
                    itemStyle: { color: "#8b5cf6" }, // violet
                },
            ],
        });
    }

    function updateBestSequence(newSequence: string) {
        if (!newSequence || newSequence === currentBestSequence) return;

        previousBestSequence = currentBestSequence;
        currentBestSequence = newSequence;

        // Find changed positions and trigger flash animation
        flashingIndices.clear();
        for (
            let i = 0;
            i < Math.max(previousBestSequence.length, newSequence.length);
            i++
        ) {
            if (previousBestSequence[i] !== newSequence[i]) {
                flashingIndices.add(i);
            }
        }
        flashingIndices = flashingIndices; // Trigger reactivity

        // Remove flash after animation duration
        setTimeout(() => {
            flashingIndices.clear();
            flashingIndices = flashingIndices; // Trigger reactivity
        }, 600);
    }

    function applyFrame(f: Frame) {
        gens.push(f.gen);
        bestFitData.push(f.summary.best_fit);
        meanFitData.push(f.summary.mean_fit);

        // Update logo data if available
        if (f.logo_counts) {
            logoData = f.logo_counts;
            updateLogoChart();
        }

        // Update best sequence visualization - get from top array if available, otherwise from best_sequence
        let bestSeq = f.best_sequence;
        if (!bestSeq && f.top && f.top.length > 0) {
            // Get the sequence with the highest fitness from the top array
            bestSeq = f.top[0].seq;
        }
        if (bestSeq) {
            updateBestSequence(bestSeq);
        }

        chartFitness?.setOption({
            xAxis: { data: gens },
            series: [
                { name: "Best", type: "line", data: bestFitData },
                { name: "Mean", type: "line", data: meanFitData },
            ],
        });

        if (f.done) {
            running = false;
            let statusMsg = `Done. Best sequence: ${f.best_sequence ?? "(see backend log)"}`;
            if (
                f.validation_fitness !== undefined &&
                f.training_fitness !== undefined
            ) {
                statusMsg += ` | Training fitness: ${f.training_fitness.toFixed(3)} | Validation fitness: ${f.validation_fitness.toFixed(3)}`;
            }
            statusText = statusMsg;
            es?.close();
            es = null;
        }
    }

    function updateLogoChart() {
        if (!chartLogo || !logoData) return;

        const positions = Array.from({ length: 20 }, (_, i) => i.toString());
        const aData: number[] = [];
        const cData: number[] = [];
        const gData: number[] = [];
        const tData: number[] = [];

        for (let i = 0; i < 20; i++) {
            const posData = logoData[i.toString()];
            if (posData) {
                aData.push(posData.A || 0);
                cData.push(posData.C || 0);
                gData.push(posData.G || 0);
                tData.push(posData.T || 0);
            } else {
                aData.push(0);
                cData.push(0);
                gData.push(0);
                tData.push(0);
            }
        }

        chartLogo.setOption({
            series: [
                { name: "A", data: aData },
                { name: "C", data: cData },
                { name: "G", data: gData },
                { name: "T", data: tData },
            ],
        });
    }

    function startRun() {
        // basic guard/validation
        if (running) return;
        if (generations < 1 || population < 10) {
            alert("Please set generations ≥ 1 and population ≥ 10");
            return;
        }

        // Validate target sequence
        targetSequenceError = validateTargetSequence(targetSequence);
        if (targetSequenceError) {
            alert(`Invalid target sequence: ${targetSequenceError}`);
            return;
        }

        resetData();
        running = true;
        statusText = "Running...";

        const url = buildStreamURL("http://localhost:8000/runs/stream", {
            lambda,
            generations,
            population_size: population,
            target_sequence: targetSequence.toUpperCase(),
        });
        es = openGAStreamURL(url, applyFrame);
    }

    function stopRun() {
        if (es) es.close();
        running = false;
        statusText = "Stopped";
    }

    onMount(() => {
        initCharts();

        // Handle window resize to ensure charts remain properly sized
        const handleResize = () => {
            chartFitness?.resize();
            chartLogo?.resize();
        };

        window.addEventListener("resize", handleResize);

        return () => {
            window.removeEventListener("resize", handleResize);
        };
    });
</script>

<div class="p-6 space-y-6">
    <!-- Logo -->
    <div class="text-center mb-8">
        <h1
            class="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent"
        >
            CRISPRight
        </h1>
        <p class="text-gray-600 text-sm mt-2">
            Genetic Algorithm for CRISPR Guide RNA Optimization
        </p>
    </div>

    <!-- Control panel -->
    <div class="bg-white rounded-xl shadow p-6">
        <!-- Target Sequence Input -->
        <div class="mb-6">
            <div class="flex flex-col items-center">
                <label
                    for="target-input"
                    class="block text-lg font-semibold text-gray-800 mb-2"
                    >Target Sequence (20 bp)</label
                >
                <input
                    id="target-input"
                    type="text"
                    maxlength="20"
                    placeholder="Enter 20 nucleotides (A, C, G, T)"
                    bind:value={targetSequence}
                    on:input={(e) => {
                        const target = e.target as HTMLInputElement;
                        targetSequence = target.value.toUpperCase();
                        targetSequenceError =
                            validateTargetSequence(targetSequence);
                    }}
                    style="text-transform: uppercase;"
                    class="border rounded-md px-4 py-3 w-80 text-center font-mono text-lg {targetSequenceError
                        ? 'border-red-500'
                        : 'border-gray-300'}"
                />
                {#if targetSequenceError}
                    <p class="text-red-500 text-sm mt-1">
                        {targetSequenceError}
                    </p>
                {/if}
            </div>
        </div>

        <!-- GA Parameters -->
        <div class="flex justify-center mb-4">
            <div class="flex flex-wrap items-end gap-4">
                <div>
                    <label
                        for="lambda-input"
                        class="block text-sm text-gray-600 mb-1"
                        >λ (off-target weight)</label
                    >
                    <input
                        id="lambda-input"
                        type="number"
                        step="0.05"
                        min="0"
                        max="2"
                        bind:value={lambda}
                        class="border rounded-md px-3 py-2 w-28"
                    />
                </div>
                <div>
                    <label
                        for="generations-input"
                        class="block text-sm text-gray-600 mb-1"
                        >Generations</label
                    >
                    <input
                        id="generations-input"
                        type="number"
                        min="1"
                        max="5000"
                        bind:value={generations}
                        class="border rounded-md px-3 py-2 w-28"
                    />
                </div>
                <div>
                    <label
                        for="population-input"
                        class="block text-sm text-gray-600 mb-1"
                        >Population</label
                    >
                    <input
                        id="population-input"
                        type="number"
                        min="10"
                        max="10000"
                        bind:value={population}
                        class="border rounded-md px-3 py-2 w-28"
                    />
                </div>

                <button
                    on:click={startRun}
                    disabled={running || targetSequenceError !== ""}
                    class="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-4 py-2 rounded-md"
                >
                    Start Run
                </button>
                <button
                    on:click={stopRun}
                    disabled={!running}
                    class="bg-gray-200 hover:bg-gray-300 text-gray-900 px-4 py-2 rounded-md"
                >
                    Stop
                </button>
            </div>
        </div>

        <!-- Status Bar -->
        <div class="text-center">
            <div class="text-sm text-gray-600">
                Status: <span class="font-medium">{statusText}</span>
            </div>
        </div>
    </div>

    <!-- Best Sequence Visualization -->
    <div class="bg-white rounded-xl shadow p-6">
        <h3 class="text-lg font-semibold text-gray-800 mb-4 text-center">
            Best Sequence {running && !currentBestSequence
                ? "(waiting for data...)"
                : ""}
        </h3>

        <!-- Target Sequence Display -->
        {#if targetSequence}
            <div class="mb-4 p-3 bg-white rounded-lg">
                <h4 class="text-sm font-medium text-gray-700 mb-2 text-center">
                    Target Sequence:
                </h4>
                <div
                    class="font-mono text-lg flex flex-wrap gap-1 justify-center"
                >
                    {#each targetSequence
                        .toUpperCase()
                        .split("") as letter, index}
                        <span
                            class="inline-block px-2 py-1 rounded bg-gray-200 text-gray-800"
                        >
                            {letter}
                        </span>
                    {/each}
                </div>
            </div>
        {/if}
        {#if currentBestSequence}
            <div class="font-mono text-lg flex flex-wrap gap-1 justify-center">
                {#each currentBestSequence.split("") as letter, index}
                    <span
                        class="inline-block px-2 py-1 rounded transition-all duration-300 nucleotide-{letter}"
                        class:flash-animation={flashingIndices.has(index)}
                        class:bg-blue-200={flashingIndices.has(index)}
                        class:text-blue-800={flashingIndices.has(index)}
                        class:font-bold={flashingIndices.has(index)}
                    >
                        {letter}
                    </span>
                {/each}
            </div>
            <div class="mt-2 text-sm text-gray-600 text-center">
                Length: {currentBestSequence.length} nucleotides
            </div>
        {:else}
            <!-- Dummy sequence with dashes -->
            <div class="font-mono text-lg flex flex-wrap gap-1 justify-center">
                {#each Array(20).fill("-") as dash, index}
                    <span
                        class="inline-block px-2 py-1 rounded bg-gray-300 text-gray-500"
                    >
                        {dash}
                    </span>
                {/each}
            </div>
            <div class="mt-2 text-sm text-gray-500 text-center">
                Waiting for genetic algorithm to start...
            </div>
        {/if}
    </div>

    <!-- Charts -->
    <div class="bg-white rounded-xl shadow p-4">
        <div id="fitness" class="w-full h-64"></div>
    </div>
    <div class="bg-white rounded-xl shadow p-4">
        <div id="logo" class="w-full h-80"></div>
    </div>
</div>

<style>
    @keyframes flash {
        0%,
        100% {
            background-color: rgb(219 234 254); /* bg-blue-200 */
            transform: scale(1);
        }
        50% {
            background-color: rgb(147 197 253); /* bg-blue-300 */
            transform: scale(1.05);
        }
    }

    .flash-animation {
        animation: flash 0.6s ease-in-out;
    }

    .nucleotide-A {
        background-color: #fef3c7;
        color: #92400e;
    } /* amber-100/amber-800 */
    .nucleotide-T {
        background-color: #ddd6fe;
        color: #5b21b6;
    } /* violet-200/violet-800 */
    .nucleotide-G {
        background-color: #d1fae5;
        color: #065f46;
    } /* emerald-200/emerald-800 */
    .nucleotide-C {
        background-color: #fed7d7;
        color: #b91c1c;
    } /* red-200/red-700 */
</style>
