import Logo from '../assets/logo.png'

const PopUpSuggestion = ({ onClose, analysisText }) => {
  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center backdrop-blur-md bg-black/40">
      <div className="flex flex-col bg-white rounded-2xl shadow-2xl p-8 w-96 text-center animate-fadeIn max-h-[80vh] overflow-y-auto">
        <div className='flex flex-col justify-center items-center'>
          <img src={Logo} alt="" className='w-[50px] h-auto mb-4'/>
          <p className="text-gray-600 mb-5"> Análise gerada por IA: </p>
          <div className="text-left text-gray-800 p-4 bg-gray-100 rounded mb-4">
            <p>{analysisText}</p>  {/* Texto da resposta LLM */}
          </div>
          
          <button
            onClick={onClose}
            className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition-all">
            Fechar
          </button>
        </div>
      </div>
    </div> 
  );
};

export default PopUpSuggestion;