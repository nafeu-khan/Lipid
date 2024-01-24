import "ag-grid-community/styles/ag-grid.css"; // Core CSS
// import "ag-grid-community/styles/ag-theme-quartz-dark.css"; // Theme
import "ag-grid-community/styles/ag-theme-quartz.css"; // Theme
import { AgGridReact } from "ag-grid-react"; // React Grid Logic
import React, { useMemo, useRef } from "react";

function AgGridTable({ rowData }) {
  const gridApi = useRef(null);
  const colDefs =
    rowData && rowData.length > 0
      ? Object.keys(rowData[0]).map((val) => ({ field: val }))
      : [];

  const defaultColDef = useMemo(
    () => ({
      filter: true, // Enable filtering on all columns
      resizable: true,
    }),
    []
  );

  const onGridReady = (params) => {
    gridApi.current = params.api;
    gridApi.current.sizeColumnsToFit();
    // Alternatively, use autoSizeColumns for specific columns
    // gridApi.current.autoSizeColumns(colDefs.map(c => c.field), false);
  };

  return (
    <div className="ag-theme-quartz" style={{ height: 500, width: "100%" }}>
      {/* The AG Grid component */}
      <AgGridReact
        onGridReady={onGridReady}
        onFirstDataRendered={onGridReady}
        rowData={rowData}
        columnDefs={colDefs}
        defaultColDef={defaultColDef}
        pagination={false}
        autoSizeStrategy={{ type: "fitCellContents" }}
      />
    </div>
  );
}

export default AgGridTable;
