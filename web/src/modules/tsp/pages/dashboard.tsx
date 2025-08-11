import { MapContainer, Marker, Polyline, TileLayer } from "react-leaflet";
import L from 'leaflet';
// import TSPNodeMarker from "../components/tspnode.marker";
import { type LatLngTuple, type Map } from "leaflet";
import useNodes from "../hooks/use-nodes";
import { useShallow } from "zustand/shallow"
import TSPNodeMarker from "../components/tspnode.marker";
import { useCallback, useRef, useState } from "react";
import { predictTSP } from "../services/api/tsp.service";
import useAxios from "../../../hooks/use-axios";
import { getGeographicMidpoint, haversine } from "../../../utils";
import DashboardSidebar from "../components/dashboard.sidebar";
import BenchmarkSidebar from "../components/benchmark.sidebar";
import type { Benchmark } from "../models/benchmark";
import useSettings from "../hooks/use-settings";
import { algorithms, groupedAlgorithms } from "../models/algorithm";


const center: LatLngTuple = [-6.923700, 106.928726]

const Dashboard = () => {
  const [
    nodes,
    routes,
    updateNode,
  ] = useNodes(useShallow(state => [
    state.nodes,
    state.routes,
    state.update,
  ]));
  const axios = useAxios();
  const [showLabel, showMarker] = useSettings(useShallow(state => [
    state.showLabel, 
    state.showMarker,
  ]));

  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([])

  const mapRef = useRef<Map>(null);

  const benchmarkAlgorithm = useCallback(async (algorithm: string) => {
    const coords = nodes.map(node => ({
      lat: node.lat,
      lon: node.lng
    }));

    const data = await predictTSP(axios, {
      model: algorithm,
      coords
    });

    const cost = data.cost;

    setBenchmarks(prev => [...prev, {
      model: algorithm,
      cost: cost
    }]);
  }, [axios, nodes])

  const handleBenchmark = async () => {
    setBenchmarks([]); // Reset benchmarks
    await Promise.allSettled(
      algorithms.map(algorithm => benchmarkAlgorithm(algorithm.key))
    );
  }

  return (
    <div className="flex size-full gap-5">
      <div className="size-full">
        <MapContainer
          ref={mapRef}
          center={center}
          zoom={16}
        >
          <TileLayer
            // attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {showMarker && nodes.map((node, index) => (
            <TSPNodeMarker
              key={index}
              node={node}
              onNodeUpdate={updateNode}
            />
          ))}

          {routes.map((route, index) => {
            // Hitung posisi polyline
            const polylinePositions = [route, ...routes.slice(index + 1)];
            if (index === routes.length - 1) {
              polylinePositions.push(routes[0]); // Tutup loop ke node pertama
            }
            return (
              <Polyline
                key={index}
                positions={polylinePositions}
                color="blue"
                weight={4}
                opacity={0.7}
              />
            );
          })}

          {showLabel && routes.map((route, index) => {
            const nextRoute = routes[(index + 1) % routes.length];
            const centerPosition = getGeographicMidpoint(route, nextRoute);

            const distance = haversine(route, nextRoute);

            // Buat icon text
            const textIcon = new L.DivIcon({
              html: `<div style="
                background: white; 
                padding: 4px 8px; 
                border-radius: 4px; 
                font-size: 12px; 
                font-weight: bold;
                border: 1px solid #ccc;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                text-align: center;
                white-space: nowrap;
              ">${distance.toFixed(2)}</div>`,
              iconSize: [80, 25],
              iconAnchor: [40, 12],
              className: 'polyline-text-marker'
            });


            return (
              <Marker
                key={index}
                position={[centerPosition[0], centerPosition[1]]}
                icon={textIcon}
                interactive={false} // Supaya tidak bisa diklik
              />
            )
          })}
        </MapContainer>
      </div>
      <DashboardSidebar
        center={center}
        algorithms={groupedAlgorithms}
        onBenchmark={handleBenchmark}
      />
      <BenchmarkSidebar
        benchmarks={benchmarks}
      />
    </div>
  );
};

export default Dashboard;
