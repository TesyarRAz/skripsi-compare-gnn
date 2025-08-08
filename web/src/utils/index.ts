import type { LatLngTuple } from "leaflet";

export const haversine = (pos1: LatLngTuple, pos2: LatLngTuple): number => {
    const R = 6371e3; // Radius of the Earth in meters
    const φ1 = (pos1[0] * Math.PI) / 180; // Convert latitude from degrees to radians
    const φ2 = (pos2[0] * Math.PI) / 180; // Convert latitude from degrees to radians
    const Δφ = ((pos2[0] - pos1[0]) * Math.PI) / 180; // Difference in latitude in radians
    const Δλ = ((pos2[1] - pos1[1]) * Math.PI) / 180; // Difference in longitude in radians

    const a =
        Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
        Math.cos(φ1) * Math.cos(φ2) *
        Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c; // Distance in meters
}

export const getGeographicMidpoint = (coord1: LatLngTuple, coord2: LatLngTuple) => {
    const [lat1, lng1] = coord1;
    const [lat2, lng2] = coord2;
    
    // Konversi ke radian
    const lat1Rad = lat1 * Math.PI / 180;
    const lng1Rad = lng1 * Math.PI / 180;
    const lat2Rad = lat2 * Math.PI / 180;
    const lng2Rad = lng2 * Math.PI / 180;
    
    const dLng = lng2Rad - lng1Rad;
    
    const bx = Math.cos(lat2Rad) * Math.cos(dLng);
    const by = Math.cos(lat2Rad) * Math.sin(dLng);
    
    const lat3Rad = Math.atan2(
      Math.sin(lat1Rad) + Math.sin(lat2Rad),
      Math.sqrt((Math.cos(lat1Rad) + bx) * (Math.cos(lat1Rad) + bx) + by * by)
    );
    const lng3Rad = lng1Rad + Math.atan2(by, Math.cos(lat1Rad) + bx);
    
    // Konversi kembali ke degrees
    const lat3 = lat3Rad * 180 / Math.PI;
    const lng3 = lng3Rad * 180 / Math.PI;
    
    return [lat3, lng3];
  }