import { useMemo, useState } from 'react'
import useNodes from '../hooks/use-nodes';
import { useShallow } from 'zustand/shallow';
import { haversine } from '../../../utils';
import NodeForm from './node.form';
import { ListSubheader, MenuItem, Select } from '@mui/material';
import { predictTSP } from '../services/api/tsp.service';
import useAxios from '../../../hooks/use-axios';
import type { LatLngTuple } from 'leaflet';

const DashboardSidebar = ({
    center,
    onBenchmark
}: {
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
    ] = useNodes(useShallow(state => [
        state.nodes,
        state.routes,
        state.update,
        state.add,
        state.remove,
        state.setRoutes,
    ]));

    const axios = useAxios();
    const [model, setModel] = useState<string>("gat/10_50");

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
    }

    return (
        <div className="hidden lg:flex flex-col h-full w-1/2 bg-gray-100 p-4 rounded-md">
            <h2 className="text-xl font-semibold">Dashboard Sidebar</h2>
            <span>Total Cost: {cost.toFixed(2)}</span>
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
                <button
                    onClick={() => addNode(center)}
                    className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 cursor-pointer"
                >
                    Add Node
                </button>

                <div className="flex flex-col gap-2">
                    <Select
                        value={model}
                        onChange={(e) => setModel(e.target.value)}
                        className="w-full"
                        inputProps={{
                            name: 'model',
                            id: 'tsp-model-select',
                        }}
                    >
                        <MenuItem value="" disabled>Select Model</MenuItem>
                        <ListSubheader>GCN</ListSubheader>
                        <MenuItem value="gcn/10_30">GCN 10-30</MenuItem>
                        <MenuItem value="gcn/10_50">GCN 10-50</MenuItem>
                        <MenuItem value="gcn/20_30">GCN 20-30</MenuItem>
                        <MenuItem value="gcn/20_50">GCN 20-50</MenuItem>
                        <ListSubheader>GAT</ListSubheader>
                        <MenuItem value="gat/10_30">GAT 10-30</MenuItem>
                        <MenuItem value="gat/10_50">GAT 10-50</MenuItem>
                        <MenuItem value="gat/20_30">GAT 20-30</MenuItem>
                        <MenuItem value="gat/20_50">GAT 20-50</MenuItem>
                        <ListSubheader>GAT V2</ListSubheader>
                        <MenuItem value="gat_v2/10_30">GAT V2 10-30</MenuItem>
                        <MenuItem value="gat_v2/10_50">GAT V2 10-50</MenuItem>
                        <MenuItem value="gat_v2/20_30">GAT V2 20-30</MenuItem>
                        <MenuItem value="gat_v2/20_50">GAT V2 20-50</MenuItem>
                    </Select>
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