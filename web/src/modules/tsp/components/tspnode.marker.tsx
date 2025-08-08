import type { LeafletEventHandlerFnMap, Marker as LeafletMarker } from "leaflet";
import { BiText } from "react-icons/bi";
import { MdClose, MdPanTool } from "react-icons/md";
import { Marker, Popup, Tooltip } from "react-leaflet";
import type { TSPNode } from "../models/tspnode";
import { useCallback, useMemo, useRef, useState, type FormEvent } from "react";

const TSPNodeMarker = ({
    node,
    onNodeUpdate
}: {
    node: TSPNode
    onNodeUpdate?: (node: TSPNode) => void;
}) => {
    const [draggable, setDraggable] = useState(false);
    const [editable, setEditable] = useState(false);
    const ref = useRef<LeafletMarker<unknown>>(null)

    const eventHandlers: LeafletEventHandlerFnMap = useMemo(() => ({
        dblclick: () => {
            setDraggable(prev => !prev)
        },
        dragend: () => {
            const marker = ref.current;
            if (marker != null) {
                const newPosition = marker.getLatLng();
                const updatedNode: TSPNode = {
                    ...node,
                    lat: newPosition.lat,
                    lng: newPosition.lng
                };
                onNodeUpdate?.(updatedNode);
            }
        }
    }), [node, onNodeUpdate])

    const handleUpdate = useCallback((e: FormEvent<HTMLInputElement>) => {
        const updatedNode: TSPNode = {
            ...node,
            name: (e.target as HTMLInputElement).value
        };
        onNodeUpdate?.(updatedNode)
    }, [node, onNodeUpdate])

    return (
        <Marker
            position={[node.lat, node.lng]}
            draggable={draggable}
            eventHandlers={eventHandlers}
            ref={ref}
        >
            
            <Popup
                autoClose={false}
                closeOnClick={false}
            >
                <div className="flex flex-col">
                    <span className="text-sm">Lat: {node.lat.toFixed(6)}</span>
                    <span className="text-sm">Lng: {node.lng.toFixed(6)}</span>
                    <span className="mt-2 text-sm">Label:
                        {
                            editable ? (
                                <input
                                    type="text"
                                    placeholder="Enter label"
                                    className="mt-2 p-1 border rounded w-full"
                                    value={node.name}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') {
                                            e.preventDefault();
        
                                            setEditable(false)
                                        }

                                        if (e.key === 'Escape') {
                                            setEditable(false);
                                        }
                                    }}
                                    onChange={handleUpdate}
                                />
                            ) : (
                                node.name
                            )
                        }</span>
                    <div className="flex gap-1">
                        <button
                            onClick={() => setDraggable(prev => !prev)}
                            className={`mt-2 px-2 py-1 rounded cursor-pointer ${draggable ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}`}
                        >
                            {draggable ? <MdClose /> : <MdPanTool />}
                        </button>
                        <button
                            onClick={() => setEditable(prev => !prev)}
                            className={`mt-2 px-2 py-1 rounded cursor-pointer ${editable ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'}`}

                        >
                            {editable ? <MdClose /> : <BiText />}
                        </button>
                    </div>
                </div>
            </Popup>
            <Tooltip
                direction="bottom"
                offset={[-10, 20]}
                opacity={1}
                permanent
            >
                <span className="text-xs">{node.name}</span>
            </Tooltip>
        </Marker>
    );
};

export default TSPNodeMarker;
