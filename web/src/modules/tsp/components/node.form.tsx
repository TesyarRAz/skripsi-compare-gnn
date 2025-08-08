import { tspNodeSchema, type TSPNode } from "../models/tspnode"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { useState } from "react";
import { BiPencil, BiTrash } from "react-icons/bi";

export interface NodeFormProps {
    node: TSPNode
    onSubmit: (node: TSPNode) => void;
    onDelete?: (node: TSPNode) => void;
}

const NodeForm = ({
    node,
    onSubmit,
    onDelete
}: NodeFormProps) => {
    const [editable, setEditable] = useState(false)

    const {
        register,
        handleSubmit,
        formState: { errors },
        getValues
    } = useForm<TSPNode>({
        resolver: zodResolver(tspNodeSchema),
        defaultValues: {
            id: node?.id || '',
            name: node?.name || '',
            lat: node?.lat || 0,
            lng: node?.lng || 0
        }
    })

    const submit = handleSubmit(() => {
        onSubmit(getValues())

        setEditable(false)
    })

    return (
        <form className="p-2 bg-white rounded shadow" onSubmit={submit}>
            {
                !editable ? (
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="text-lg font-medium" onDoubleClick={() => setEditable(true)}>
                            {node.name}
                        </h3>

                        <div className="flex items-center gap-2">
                            <button
                                type="button"
                                className="text-blue-500 hover:underline cursor-pointer"
                                onClick={() => setEditable(true)}
                            >
                                <BiPencil />
                            </button>
                            <button
                                type="button"
                                className="text-red-500 hover:underline cursor-pointer"
                                onClick={() => onDelete?.(node)} // Assuming empty name means delete
                            >
                                <BiTrash />
                            </button>
                        </div>
                    </div>
                ) : (
                    <>
                        <input
                            type="text"
                            {...register("name")}
                            className="w-full p-2 border rounded mb-2"
                            placeholder="Node Name"
                        />
                        <span className="text-red-500">
                            {errors.name?.message}
                        </span>
                    </>
                )
            }
            <p>Lat: {node.lat.toFixed(6)}</p>
            <p>Lng: {node.lng.toFixed(6)}</p>
        </form>
    )
}

export default NodeForm;