import { Select } from "antd";
import { useDispatch } from "react-redux";
import {
  changeNumOfComp,
} from "../Slices/StructureAnalysisSlice";

const SelectCompositionType = ({ type, setType, setLipidInput }) => {
  const dispatch = useDispatch();
  return (
    <div className="space-y-1.5">
      <label htmlFor="numOfComp" className=" text-gray-700 font-medium">
        Number of Compositions
      </label>
      <Select
        defaultValue={type}
        style={{ width: "100%" }}
        size="large"
        onChange={(e) => {
          if (e === "single") {
            setLipidInput([{ name: "", percentage: 100 }]);
          } else {
            setLipidInput([
              { name: "", percentage: 0 },
              { name: "", percentage: 0 },
            ]);
          }
          dispatch(changeNumOfComp(e));
          setType(e);
        }}
        options={[
          { value: "single", label: "Single" },
          // { value: "multiple", label: "Multiple" },
        ]}
      />
    </div>
  );
};

export default SelectCompositionType;
