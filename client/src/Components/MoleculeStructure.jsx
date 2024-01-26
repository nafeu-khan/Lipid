import React from "react";
import { useSelector } from "react-redux";
import ActualPredicted from "./ActualPredicted";
import ChartComponent from "./ChartComponent";
import GraphTable from "./GraphTable";

function MoleculeStructure() {
  const { data, showTable, lipid } = useSelector((state) => state.lipid);

  return (
    <>
      <ActualPredicted />
      <div className="w-full h-full flex items-center">
        <ChartComponent
          graph_data={data && data.predicted && [data.predicted[lipid[0].name]]}
        />
        <div className={`${showTable ? "w-[600px] px-2" : "w-0"}`}>
          <GraphTable />
        </div>
      </div>
    </>
  );
}

export default MoleculeStructure;
