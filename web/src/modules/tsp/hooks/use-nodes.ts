import { create } from "zustand";
import type { TSPNode } from "../models/tspnode";
import type { LatLngTuple } from "leaflet";

export interface NodesState {
    nodes: TSPNode[];
    routes: LatLngTuple[];
    update: (updated: TSPNode) => void;
    add: (center: LatLngTuple) => void;
    remove: (id: string) => void;
    setRoutes: (routes: LatLngTuple[]) => void;
}

const useNodes = create<NodesState>((set) => ({
    nodes: [],
    routes: [],
    update(updated: TSPNode) {
        set((state) => ({
            nodes: state.nodes.map((node) =>
                node.id === updated.id ? { ...node, ...updated } : node
            ),
            routes: [],
        }));
    },
    add(center: LatLngTuple) {
        set((state) => ({
            nodes: [...state.nodes, {
                lat: center[0] + Math.random() * 0.01, // Randomly offset latitude
                lng: center[1] + Math.random() * 0.01, // Randomly offset longitude
                id: Date.now().toString(), // Unique ID based on timestamp
                name: `Node ${state.nodes.length + 1}` // Simple naming convention
            }],
            routes: []
        }));
    },
    remove(id: string) {
        set((state) => ({
            nodes: state.nodes.filter((node) => node.id !== id),
            routes: [],
        }));
    },
    setRoutes(routes: LatLngTuple[]) {
        set(() => ({
            routes: routes,
        }));
    }
}))

export default useNodes;