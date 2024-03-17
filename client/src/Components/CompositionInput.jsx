import { Input, InputNumber } from "antd";

const CompositionInput = ({
  id,
  showPercentage,
  name,
  percentage,
  onCompositionChange,
}) => (
  <div className="flex items-center gap-4">
    <div className="w-full">
      <label className="text-gray-800 font-bold">
        Lipid Composition Name{showPercentage ? `-${id}` : ""}
      </label>
      <Input
        value={name}
        size="large"
        onChange={(e) => onCompositionChange(id, "name", e.target.value)}
      />
    </div>
    {showPercentage && (
      <div>
        <label className="text-gray-800">Percentage</label>
        <InputNumber
          size="large"
          value={percentage}
          className="mt-1.5 w-full"
          onChange={(value) => onCompositionChange(id, "percentage", value)}
          defaultValue={0}
        />
      </div>
    )}
  </div>
);

export default CompositionInput;
