import torch
import torch.nn.functional as F
from fastapi import HTTPException
import utils
import model
import algorithm

def predict_only(net, sample, device):
    # Set evaluation mode
    net.eval()

    with torch.no_grad():
        # Convert batch to torch tensors
        x_edges = torch.tensor(sample.edges, dtype=torch.long, device=device)
        x_edges_values = torch.tensor(sample.edges_values, dtype=torch.float32, device=device)
        x_nodes = torch.tensor(sample.nodes, dtype=torch.long, device=device)
        x_nodes_coord = torch.tensor(sample.nodes_coord, dtype=torch.float32, device=device)

        # Forward pass
        y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges=None, edge_cw=None)

        # Prepare for plotting
        x_coord = x_nodes_coord.cpu().numpy()
        W = x_edges.cpu().numpy()
        W_val = x_edges_values.cpu().numpy()

        y = F.softmax(y_preds, dim=3)  # B x V x V x voc_edges
        y_bins = y.argmax(dim=3)      # B x V x V (binary edge predictions)
        y_probs = y[:, :, :, 1]       # B x V x V (edge existence probs)
        W_sol_probs = y_probs[0].cpu().numpy()

        # ==== Simple Beamsearch ====
        y_probs_cpu = y_probs[0].cpu()             # (V, V)
        W_val_cpu = x_edges_values[0].cpu().numpy()
        tour_mask, score = utils.simple_beamsearch(y_probs_cpu, W_val_cpu, beam_size=5)

        return tour_mask, score


def predict_with_coords(models, device, tour: model.PredictCoordTour):
    model_name = tour.model
    sample = utils.generate_sample_from_coords(
        coords=[(coord.lat, coord.lon) for coord in tour.coords],
    )

    if model_name not in models:
        if model_name == 'ant_colony':
            coords = [(coord.lat, coord.lon) for coord in tour.coords]
            tour_mask, cost = algorithm.ant_colony(coords)
        elif model_name == 'held_karp':
            if len(tour.coords) > 20:
                raise HTTPException(status_code=400, detail="Held-Karp algorithm is not suitable for more than 20 nodes.")
            coords = [(coord.lat, coord.lon) for coord in tour.coords]
            tour_mask, cost = algorithm.held_karp(coords)
        else:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found.")
    
    else:
        model_instance = models[model_name]

        tour_mask, _ = predict_only(model_instance, sample, device)

        cost = utils.path_cost(tour_mask, sample.edges_values[0])

    return {
        "tour_mask": tour_mask,
        # "edge_values": sample.edges_values,
        "cost": cost,
    }