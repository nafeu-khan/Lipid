import { Input, InputNumber } from "antd";

const LipidInputForm = ({ lipidInput, handleInputChange, type }) => (
  <>
    {lipidInput.map((item, index) => {
      let maxPercentage = 100;
      if (lipidInput.length === 2) {
        const otherIndex = index === 0 ? 1 : 0;
        maxPercentage = 100 - lipidInput[otherIndex].percentage;
      }

      return (
        <div key={index} className="flex gap-4 mt-3">
          {/* Lipid Name Input */}
          <div>
            <label className="text-gray-700/80">Lipid Name</label>
            <Input
              size="large"
              placeholder="POPC"
              value={item.name}
              onChange={(e) => handleInputChange(index, "name", e.target.value)}
            />
          </div>
          {/* Percentage Input */}
          <div>
            <label className=" text-gray-700/80">Percentage</label>
            <div className="flex items-center gap-1">
              <InputNumber
                size="large"
                min={0}
                max={maxPercentage}
                className="w-16"
                disabled={type === "single"}
                value={item.percentage}
                onChange={(e) => handleInputChange(index, "percentage", e)}
              />
              <span className="text-sm text-gray-700/80">%</span>
            </div>
          </div>
        </div>
      );
    })}
  </>
);

export default LipidInputForm;
