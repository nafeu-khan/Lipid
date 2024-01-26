import React from "react";
import { useSelector } from "react-redux";
import ActvspredGraph from "../Components/ActvspredGraph";

const PredictionKappa = () => {
  const { data, lipid } = useSelector((state) => state.lipid);

  return (
    <div className="w-full h-full flex flex-col items-center p-2">
      <h1 className="bg-violet-500 text-gray-100 mt-2 p-2 px-4 text-lg rounded shadow font-mono">
        Prediction Value:{" "}
        <span className="text-white font-semibold">{data.pred}</span>
      </h1>
      {data.graph ? (
        <img
          className="mt-8"
          src={`data:image/png;base64,${data.graph}`}
          alt=""
        />
      ) : (
        <h1 className="mt-4">No predicted data found.</h1>
      )}
      <div className="grid grid-cols-3 gap-8 w-full mt-8">
        <div className="w-full">
          <h3 className="text-center font-medium mb-2 text-lg">
            Actual vs Predicted for - {lipid[0]?.name}
          </h3>
          <ActvspredGraph />
        </div>
        <div></div>
      </div>
    </div>
  );
};

export default PredictionKappa;
