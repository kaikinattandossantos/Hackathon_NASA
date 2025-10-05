const FilterCard = ({title, description, colorClass, onClick}) => {
  
  return (
    <button onClick={onClick}>
      <div className={`shadow-[inset_0px_5px_15px_-3px_rgba(0,_0,_0,_0.2)] flex flex-row col-span-1 ${colorClass} backdrop-blur-lg h-auto border-2 rounded-md border-ocean-light/10
      hover:bg-white/30 transition-colors ease-in-out duration-800>`}>
          <div className="flex flex-col justify-center items-center p-4">
              <p className="font-bold text-ocean-blue"> {title} </p>
              <p className="text-[12px] text-ocean-blue"> {description} </p>
          </div>

      </div>
    </button>
    
  );
};

export default FilterCard;