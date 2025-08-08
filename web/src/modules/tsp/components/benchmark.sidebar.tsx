import React, { useMemo } from 'react'
import type { Benchmark } from '../models/tspnode'

const BenchmarkSidebar = ({
  benchmarks
}: {
  benchmarks: Benchmark[]
}) => {
  const sortedBenchmarks = useMemo(() => [...benchmarks].sort((a, b) => a.cost - b.cost), [benchmarks])

  return (
    <div className="flex flex-col gap-4 p-4">
      <h2 className="text-lg font-semibold">Benchmark Algorithms</h2>
      <p className="text-sm text-gray-600">
        Compare the performance of different TSP algorithms.
      </p>
      <table>
        <thead>
          <tr>
            <th className="px-4 py-2 text-left">Algorithm</th>
            <th className="px-4 py-2 text-right">Cost</th>
          </tr>
        </thead>
        <tbody>
          {sortedBenchmarks.map((benchmark, index) => (
            <tr key={index} className="border-b">
              <td className="px-4 py-2">{benchmark.model}</td>
              <td className="px-4 py-2 text-right">{benchmark.cost.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default BenchmarkSidebar