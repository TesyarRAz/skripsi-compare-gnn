import { tspNodeSchema, type TSPNode } from "../models/tspnode"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"

export interface NodeFormProps {
    node?: TSPNode | null
    handleCreateOrUpdate: (node: TSPNode, option: string) => void
    closeModal: () => void
}

const NodeForm = ({
    node,
    handleCreateOrUpdate,
    closeModal
}: NodeFormProps) => {
    const {
        register,
        handleSubmit,
        formState: {errors}
    } = useForm<TSPNode>({
        resolver: zodResolver(tspNodeSchema),
        defaultValues: {
            id: node?.id || '',
            name: node?.name || '',
            lat: node?.lat || 0,
            lng: node?.lng || 0
        }
    })

    const onSubmit = handleSubmit(async (values: TSPNode) => {
        
    })

    return (
        <form onSubmit={onSubmit} className="space-y-4">
            
        </form>
    )
}