import React from 'react'

function Header() {
  return (
    <div className='flex items-center justify-center border-b py-2 shadow bg-[whitesmoke] cursor-pointer' onClick={() => window.location.href='/'}>
        <img src="logo.gif" alt="" className='w-12 h-12' />
        <h1 className='font-semibold text-xl tracking-wider font-serif'>Lipid Membrane</h1>
    </div>
  )
}

export default Header