import { MapContainer, TileLayer } from "react-leaflet";
import TSPNodeMarker from "../components/tspnode.marker";
import type { LatLngTuple } from "leaflet";
import { useState } from "react";
import type { TSPNode } from "../models/tspnode";
import { Container } from "@mui/material";

const center: LatLngTuple = [-6.923700, 106.928726]

const Dashboard = () => {
  const [nodes, setNodes] = useState<TSPNode[]>([]);

  const handleAddNode = () => {
    // Function to handle adding a new node
    const newNode: TSPNode = {
      lat: center[0] + Math.random() * 0.01, // Randomly offset latitude
      lng: center[1] + Math.random() * 0.01, // Randomly offset longitude
      id: Date.now().toString(), // Unique ID based on timestamp
      name: `Node ${nodes.length + 1}` // Simple naming convention
    }
    setNodes(prevNodes => [...prevNodes, newNode]);
  }

  const handleNodeUpdate = (updatedNode: TSPNode) => {
    // Function to handle updating a node's position
    setNodes(prevNodes => prevNodes.map(node => 
      node.id === updatedNode.id ? updatedNode : node
    ));
  }

  return (
    <Container className="flex size-full gap-5">
      <div className="size-full">
        <MapContainer
          center={center}
          zoom={13}
        >
          <TileLayer
            // attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {nodes.map((node, index) => (
            <TSPNodeMarker
              key={index}
              node={node}
              onNodeUpdate={handleNodeUpdate}
              />
          ))}
        </MapContainer>
      </div>
      <div className="hidden lg:block w-1/2 bg-gray-100 p-4 rounded-md">
        <h2 className="text-xl font-semibold mb-4">Dashboard Sidebar</h2>
        <button
          onClick={handleAddNode}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 cursor-pointer"
        >
          Add Node
        </button>
      </div>
    </Container>
  );
};

export default Dashboard;
