import React from 'react';

const RankCard = ({ rank, name, score, onClick, isSelected }) => {
  return (
    <div
      className={`flex justify-between items-center mx-2 py-3 px-2 border-b-2 border-gray-700/50 cursor-pointer transition-all ease-in-out duration-300 ${isSelected ? 'bg-cyan-800/50' : 'hover:bg-gray-700/30'}`}
      onClick={onClick}
    >
      <div className='flex flex-row items-center'>
        <div className='inline-block bg-purple-labubu rounded-lg p-2 h-10 w-10 border-2 border-white/40 flex-shrink-0'>
          <p className='flex justify-center items-center text-space-background font-extrabold text-sm'>{rank}</p>
        </div>
        <p className='ml-4 font-semibold text-white text-sm'>{name}</p>
      </div>
      <p className='font-normal mr-2 text-gray-400 text-sm'>{score.toFixed(3)}</p>
    </div>
  );
};

export default RankCard;
