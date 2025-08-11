import { useEffect, useMemo, useState } from 'react'
import useNodes from '../hooks/use-nodes';
import { useShallow } from 'zustand/shallow';
import { haversine } from '../../../utils';
import NodeForm from './node.form';
import { FormControl, InputLabel, ListSubheader, MenuItem, Select } from '@mui/material';
import { predictTSP } from '../services/api/tsp.service';
import useAxios from '../../../hooks/use-axios';
import type { LatLngTuple } from 'leaflet';
import useSettings from '../hooks/use-settings';
import { AxiosError } from 'axios';
import type { Algorithm } from '../models/algorithm';
import { schenarios } from '../models/tspnode';

const DashboardSidebar = ({
    algorithms,
    center,
    onBenchmark
}: {
    algorithms: Record<string, Algorithm[]>;
    center: LatLngTuple;
    onBenchmark: () => void;
}) => {
    const [
        nodes,
        routes,
        updateNode,
        addNode,
        removeNode,
        setRoutes,
        clearNodes,
        setNodes,
        noiseNodes,
    ] = useNodes(useShallow(state => [
        state.nodes,
        state.routes,
        state.update,
        state.add,
        state.remove,
        state.setRoutes,
        state.clearNodes,
        state.setNodes,
        state.noiseNodes
    ]));

    const axios = useAxios();
    const [
        showLabel,
        showMarker,
        setShowLabel,
        setShowMarker,
    ] = useSettings(useShallow(state => [
        state.showLabel,
        state.showMarker,
        state.setShowLabel,
        state.setShowMarker
    ]));
    const [model, setModel] = useState<string>(Object.entries(algorithms)[0]?.[1][0]?.key || "");
    const [schenarioIndex, setSchenarioIndex] = useState<number | null>(null);

    useEffect(() => {
        if (schenarioIndex !== null && schenarios[schenarioIndex]) {
            setNodes(schenarios[schenarioIndex])
        }
    }, [schenarioIndex, setNodes]);

    const cost = useMemo(() => {
        if (routes.length === 0) return 0;
        return routes.reduce((total, route, index) => {
            const nextRoute = routes[(index + 1) % routes.length];
            return total + haversine(route, nextRoute);
        }, 0);
    }, [routes])

    const handlePredict = async () => {
        if (!model) {
            alert("Please select a model");
            return;
        }

        if (nodes.length < 2) {
            alert("Please add at least two nodes to predict the TSP route");
            return;
        }

        const coords = nodes.map(node => ({
            lat: node.lat,
            lon: node.lng
        }));

        try {
            const data = await predictTSP(axios, {
                model,
                coords
            })

            if (data.tour_mask.length === 0) {
                alert("No route found. Please check your nodes and model selection.");
                return;
            }

            const routes = data.tour_mask.map(index => nodes[index]);
            setRoutes(routes.map(node => [node.lat, node.lng]));
        } catch (error: AxiosError | unknown) {
            if (error instanceof AxiosError) {
                alert(`Error: ${error.response?.data?.detail || error.message}`);
            } else {
                alert("An unexpected error occurred while predicting the TSP route.");
            }
        }
    }

    const handleClearNode = () => {
        clearNodes();
        setSchenarioIndex(null);
    }

    const handleAddNode = () => {
        addNode(center);
        setSchenarioIndex(null);
    }

    return (
        <div className="hidden lg:flex flex-col h-full w-1/2 bg-gray-100 p-4 rounded-md">
            <h2 className="text-xl font-semibold">Dashboard Sidebar</h2>
            <span>Total Cost: {cost.toFixed(2)}</span>
            <div className="flex items-center gap-2">
                <input
                    type="checkbox"
                    checked={showMarker}
                    onChange={(e) => setShowMarker(e.target.checked)}
                />
                <label>Show Marker</label>
                <input
                    type="checkbox"
                    checked={showLabel}
                    onChange={(e) => setShowLabel(e.target.checked)}
                />
                <label>Show Label</label>
            </div>
            <div className="flex-1 flex flex-col gap-5 py-10">
                <div className="overflow-y-scroll flex-1 px-2">
                    <div className="flex flex-col gap-1 flex-1/2 max-h-80">
                        {
                            nodes.map((node) => (
                                <NodeForm
                                    key={node.id}
                                    node={node}
                                    onSubmit={(node) => {
                                        updateNode(node);
                                    }}
                                    onDelete={(node) => {
                                        removeNode(node.id);
                                    }}
                                />
                            ))
                        }
                    </div>
                </div>

                <div className="flex flex-col gap-2">
                    <Select
                        label="Select Schenario"
                        value={schenarioIndex ?? ""}
                        onChange={(e) => setSchenarioIndex(e.target.value)}
                        className='w-full'
                    >
                        <MenuItem value="" disabled>Select Schenario</MenuItem>
                        {schenarios.map((_, index) => (
                            <MenuItem
                                key={index}
                                value={index}
                            >
                                Schenario {index + 1}
                            </MenuItem>
                        ))}
                    </Select>
                </div>

                <button
                    onClick={() => handleAddNode()}
                    className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 cursor-pointer"
                >
                    Add Node
                </button>
                <button
                    onClick={() => noiseNodes()}
                    className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600 cursor-pointer"
                >
                    Add Noise to Nodes
                </button>
                <button
                    onClick={() => handleClearNode()}
                    className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 cursor-pointer"
                >
                    Clear Nodes
                </button>


                <div className="flex flex-col gap-2">
                    <FormControl
                    >
                        <InputLabel id="select-model-label">Select Model</InputLabel>
                        <Select
                            labelId='select-model-label'
                            id="select-model"
                            value={model}
                            onChange={(e) => setModel(e.target.value)}
                        >
                            <MenuItem value="" disabled>Select Model</MenuItem>
                            {Object.entries(algorithms).map(([group, algos]) => [
                                <ListSubheader>{group}</ListSubheader>,
                                algos.map((algo) => (
                                    <MenuItem key={algo.key} value={algo.key}>
                                        {algo.name}
                                    </MenuItem>
                                ))
                            ])}
                        </Select>
                    </FormControl>
                    <button
                        type="button"
                        onClick={handlePredict}
                        className="mt-4 bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 w-full cursor-pointer"
                    >
                        Predict TSP Route
                    </button>
                    <button
                        type="button"
                        onClick={onBenchmark}
                        className="mt-2 bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600 w-full cursor-pointer"
                    >
                        Benchmark Model
                    </button>
                </div>
            </div>
        </div>
    )
}

export default DashboardSidebar