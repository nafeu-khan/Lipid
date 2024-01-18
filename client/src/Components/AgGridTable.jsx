import "ag-grid-community/styles/ag-grid.css"; // Core CSS
// import "ag-grid-community/styles/ag-theme-quartz-dark.css"; // Theme
import "ag-grid-community/styles/ag-theme-quartz.css"; // Theme
import { AgGridReact } from "ag-grid-react"; // React Grid Logic
import React, { useMemo } from "react";

function AgGridTable({ rowData }) {
  const colDefs =
    rowData && rowData.length > 0
      ? Object.keys(rowData[0]).map((val) => ({ field: val }))
      : [];

  const defaultColDef = useMemo(
    () => ({
      filter: true, // Enable filtering on all columns
      resizable: true
    }),
    []
  );

  return (
    <div
      className="ag-theme-quartz"
      style={{ height: 500, width: "100%" }}
    >
      {/* The AG Grid component */}
      <AgGridReact
        rowData={rowData}
        columnDefs={colDefs}
        defaultColDef={defaultColDef}
        pagination={true}
      />
    </div>
  );
}

export default AgGridTable;
