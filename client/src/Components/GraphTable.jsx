import React, { useEffect, useState } from "react";
import { useSelector } from "react-redux";
import AgGridTable from "./AgGridTable";

function GraphTable() {
  const edges = useSelector((state) => state.structure.data.edge_table);
  const [rowData, setRowData] = useState([]);

  useEffect(() => {
    if(!edges) return;
    const nodeEdgesMap = {};
    edges.forEach((edge) => {
      edge.forEach((node) => {
        if (!nodeEdgesMap[node]) {
          nodeEdgesMap[node] = new Set();
        }
        nodeEdgesMap[node].add(edge.join(" - "));
      });
    });

    const formattedRowData = Object.entries(nodeEdgesMap).map(
      ([node, edgeSet]) => ({
        node: node,
        edges: Array.from(edgeSet).join(", "),
      })
    );

    setRowData(formattedRowData);
  }, [edges]);

  return <AgGridTable rowData={rowData} />;
}

export default GraphTable;
